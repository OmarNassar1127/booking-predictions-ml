"""
Laravel Callback Webhooks

This module sends updated predictions back to Laravel when cascade re-prediction
occurs. It includes HMAC signature verification for security.

Usage:
    webhook_client = get_webhook_client()
    await webhook_client.send_predictions_updated(affected_bookings)
"""

import hmac
import hashlib
import json
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import aiohttp
from aiohttp import ClientTimeout

from ..utils.logger import logger
from ..utils.config_loader import config


class LaravelWebhookClient:
    """Client for sending webhooks to Laravel"""

    def __init__(
        self,
        callback_url: Optional[str] = None,
        secret_key: Optional[str] = None,
        enabled: bool = True,
        timeout_seconds: int = 10
    ):
        """
        Initialize webhook client

        Args:
            callback_url: Laravel callback URL (from config if not provided)
            secret_key: HMAC secret for signature (from config if not provided)
            enabled: Whether webhooks are enabled (from config if not provided)
            timeout_seconds: Request timeout in seconds
        """
        self.callback_url = callback_url or config.get('webhooks.callback_url')
        self.secret_key = secret_key or config.get('webhooks.secret_key', '')
        self.enabled = enabled if enabled is not None else config.get('webhooks.enabled', True)
        self.timeout = ClientTimeout(total=timeout_seconds)

        if self.enabled and not self.callback_url:
            logger.warning("Webhooks enabled but no callback_url configured")
            self.enabled = False

    def _generate_signature(self, payload: str) -> str:
        """
        Generate HMAC SHA256 signature for payload

        Args:
            payload: JSON string of the payload

        Returns:
            Hex-encoded HMAC signature
        """
        if not self.secret_key:
            return ""

        return hmac.new(
            key=self.secret_key.encode('utf-8'),
            msg=payload.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()

    async def send_predictions_updated(
        self,
        event_type: str,
        affected_bookings: List[Dict],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Send predictions updated callback to Laravel

        Args:
            event_type: Type of event that triggered update (booking.started, booking.ended, etc.)
            affected_bookings: List of bookings with updated predictions
            metadata: Optional additional data

        Returns:
            True if webhook sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Webhooks disabled, skipping callback")
            return False

        if not affected_bookings:
            logger.debug("No affected bookings, skipping callback")
            return True

        # Build payload
        payload_data = {
            "event": "predictions.updated",
            "triggered_by": event_type,
            "timestamp": datetime.now().isoformat(),
            "affected_bookings": affected_bookings,
            "metadata": metadata or {}
        }

        # Convert to JSON
        payload_json = json.dumps(payload_data, separators=(',', ':'))

        # Generate signature
        signature = self._generate_signature(payload_json)

        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'X-ML-Signature': signature,
            'X-ML-Event': 'predictions.updated',
            'X-ML-Timestamp': payload_data['timestamp'],
            'User-Agent': 'BatteryPredictionML/2.0'
        }

        # Send webhook
        try:
            logger.info(f"Sending predictions.updated callback: {len(affected_bookings)} bookings")

            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    self.callback_url,
                    json=payload_data,
                    headers=headers
                ) as response:
                    response_text = await response.text()

                    if response.status == 200:
                        logger.info(f"✓ Callback successful: {response.status}")
                        return True
                    else:
                        logger.warning(f"Callback failed: {response.status} - {response_text}")
                        return False

        except asyncio.TimeoutError:
            logger.error(f"Callback timeout after {self.timeout.total}s")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"Callback client error: {e}")
            return False
        except Exception as e:
            logger.error(f"Callback unexpected error: {e}")
            return False

    async def send_pattern_updated(
        self,
        vehicle_id: int,
        pattern_type: str,
        old_value: float,
        new_value: float,
        confidence: str
    ) -> bool:
        """
        Notify Laravel that vehicle patterns were updated

        Args:
            vehicle_id: Vehicle ID
            pattern_type: Type of pattern (drain_rate, charging_frequency, etc.)
            old_value: Previous value
            new_value: New value
            confidence: Confidence level (low, medium, high)

        Returns:
            True if webhook sent successfully
        """
        if not self.enabled:
            return False

        payload_data = {
            "event": "pattern.updated",
            "timestamp": datetime.now().isoformat(),
            "vehicle_id": vehicle_id,
            "pattern_type": pattern_type,
            "old_value": old_value,
            "new_value": new_value,
            "confidence": confidence
        }

        payload_json = json.dumps(payload_data, separators=(',', ':'))
        signature = self._generate_signature(payload_json)

        headers = {
            'Content-Type': 'application/json',
            'X-ML-Signature': signature,
            'X-ML-Event': 'pattern.updated',
            'User-Agent': 'BatteryPredictionML/2.0'
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    self.callback_url,
                    json=payload_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Pattern update callback successful for vehicle {vehicle_id}")
                        return True
                    else:
                        logger.warning(f"Pattern update callback failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Pattern callback error: {e}")
            return False

    def send_predictions_updated_sync(
        self,
        event_type: str,
        affected_bookings: List[Dict],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Synchronous wrapper for send_predictions_updated

        Use this when calling from sync context (FastAPI background tasks)

        Args:
            event_type: Type of event
            affected_bookings: Updated bookings
            metadata: Optional metadata

        Returns:
            True if successful
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.send_predictions_updated(event_type, affected_bookings, metadata)
        )

    async def test_connection(self) -> bool:
        """
        Test webhook connection to Laravel

        Returns:
            True if Laravel responds successfully
        """
        if not self.enabled or not self.callback_url:
            logger.warning("Webhooks not configured, cannot test")
            return False

        test_payload = {
            "event": "test.connection",
            "timestamp": datetime.now().isoformat(),
            "message": "Testing webhook connection from ML system"
        }

        payload_json = json.dumps(test_payload, separators=(',', ':'))
        signature = self._generate_signature(payload_json)

        headers = {
            'Content-Type': 'application/json',
            'X-ML-Signature': signature,
            'X-ML-Event': 'test.connection',
            'User-Agent': 'BatteryPredictionML/2.0'
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    self.callback_url,
                    json=test_payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        logger.info("✓ Webhook connection test successful")
                        return True
                    else:
                        logger.warning(f"Webhook test failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Webhook test error: {e}")
            return False


# Global instance
_webhook_client: Optional[LaravelWebhookClient] = None


def get_webhook_client() -> LaravelWebhookClient:
    """Get global webhook client instance"""
    global _webhook_client
    if _webhook_client is None:
        _webhook_client = LaravelWebhookClient()
    return _webhook_client


def configure_webhook_client(
    callback_url: str,
    secret_key: str,
    enabled: bool = True
):
    """
    Configure global webhook client

    Args:
        callback_url: Laravel callback URL
        secret_key: HMAC secret key
        enabled: Whether to enable webhooks
    """
    global _webhook_client
    _webhook_client = LaravelWebhookClient(
        callback_url=callback_url,
        secret_key=secret_key,
        enabled=enabled
    )
    logger.info(f"Webhook client configured: {callback_url}")
