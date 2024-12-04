'''
# email_service.py
from builtins import ValueError, dict, str
from settings.config import settings
from app.utils.smtp_connection import SMTPClient
from app.utils.template_manager import TemplateManager
from app.models.user_model import User

class EmailService:
    def __init__(self, template_manager: TemplateManager):
        self.smtp_client = SMTPClient(
            server=settings.smtp_server,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password
        )
        self.template_manager = template_manager

    async def send_user_email(self, user_data: dict, email_type: str):
        subject_map = {
            'email_verification': "Verify Your Account",
            'password_reset': "Password Reset Instructions",
            'account_locked': "Account Locked Notification"
        }

        if email_type not in subject_map:
            raise ValueError("Invalid email type")

        html_content = self.template_manager.render_template(email_type, **user_data)
        self.smtp_client.send_email(subject_map[email_type], html_content, user_data['email'])

    async def send_verification_email(self, user: User):
        verification_url = f"{settings.server_base_url}verify-email/{user.id}/{user.verification_token}"
        await self.send_user_email({
            "name": user.first_name,
            "verification_url": verification_url,
            "email": user.email
        }, 'email_verification')
        '''

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.email_service import EmailService
from app.utils.template_manager import TemplateManager
from app.models.user_model import User


@pytest.fixture
def mock_template_manager():
    """Provides a mock template manager."""
    template_manager = MagicMock(spec=TemplateManager)
    template_manager.render_template.return_value = "<html>Email Content</html>"
    return template_manager


@pytest.fixture
def mock_smtp_client():
    """Mock the SMTP client."""
    with patch("app.utils.smtp_connection.SMTPClient") as MockSMTPClient:
        instance = MockSMTPClient.return_value
        instance.send_email = MagicMock()
        yield instance


@pytest.fixture
def email_service(mock_template_manager):
    """Provides the EmailService instance with mocked dependencies."""
    return EmailService(template_manager=mock_template_manager)


@pytest.fixture
def mock_user():
    """Provides a mock user object."""
    return User(
        id="123e4567-e89b-12d3-a456-426614174000",
        first_name="Test",
        email="test@example.com",
        verification_token="mock_token"
    )


@pytest.mark.asyncio
async def test_send_user_email_success(email_service, mock_smtp_client):
    """Test that send_user_email sends an email successfully."""
    user_data = {
        "name": "Test User",
        "email": "test@example.com",
        "verification_url": "http://example.com/verify?token=abc123"
    }
    await email_service.send_user_email(user_data, 'email_verification')

    email_service.template_manager.render_template.assert_called_once_with(
        'email_verification',
        **user_data
    )
    mock_smtp_client.send_email.assert_called_once_with(
        "Verify Your Account",
        "<html>Email Content</html>",
        "test@example.com"
    )


@pytest.mark.asyncio
async def test_send_user_email_invalid_type(email_service):
    """Test that sending an email with an invalid type raises a ValueError."""
    user_data = {
        "name": "Test User",
        "email": "test@example.com",
        "verification_url": "http://example.com/verify?token=abc123"
    }
    with pytest.raises(ValueError, match="Invalid email type"):
        await email_service.send_user_email(user_data, 'invalid_type')


@pytest.mark.asyncio
async def test_send_verification_email_success(email_service, mock_smtp_client, mock_user):
    """Test that send_verification_email sends a verification email successfully."""
    await email_service.send_verification_email(mock_user)

    email_service.template_manager.render_template.assert_called_once_with(
        'email_verification',
        name=mock_user.first_name,
        verification_url=f"http://localhost/verify-email/{mock_user.id}/{mock_user.verification_token}",
        email=mock_user.email
    )
    mock_smtp_client.send_email.assert_called_once_with(
        "Verify Your Account",
        "<html>Email Content</html>",
        mock_user.email
    )


@pytest.mark.asyncio
async def test_send_verification_email_smtp_failure(email_service, mock_smtp_client, mock_user):
    """Test handling of SMTP errors during send_verification_email."""
    mock_smtp_client.send_email.side_effect = Exception("SMTP error")

    with pytest.raises(Exception, match="SMTP error"):
        await email_service.send_verification_email(mock_user)

    email_service.template_manager.render_template.assert_called_once()
    mock_smtp_client.send_email.assert_called_once()
