"""Email service for sending invitations, verification emails, etc."""

import os
import logging
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via SMTP."""

    def __init__(self):
        """Initialize email service with SMTP configuration from environment."""
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@example.com")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

        # Check if SMTP is configured
        self.enabled = bool(self.smtp_user and self.smtp_password)

        if not self.enabled:
            logger.warning("[Email] SMTP not configured. Email sending disabled.")
            logger.warning("[Email] Set SMTP_USER and SMTP_PASSWORD in .env to enable email.")
        else:
            logger.info(f"[Email] Service initialized: {self.smtp_user}@{self.smtp_host}:{self.smtp_port}")

    def _send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        """
        Send an email via SMTP.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML email body
            text_body: Plain text email body (fallback)

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning(f"[Email] Skipping email to {to_email} (SMTP not configured)")
            logger.info(f"[Email] Would have sent: {subject}")
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = to_email

            # Add text and HTML parts
            if text_body:
                part1 = MIMEText(text_body, "plain")
                msg.attach(part1)

            part2 = MIMEText(html_body, "html")
            msg.attach(part2)

            # Send email via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"[Email] Successfully sent email to {to_email}: {subject}")
            return True

        except Exception as e:
            logger.error(f"[Email] Failed to send email to {to_email}: {e}")
            return False

    def send_invitation_email(
        self,
        to_email: str,
        token: str,
        inviter_name: str,
        entity_type: str,
        entity_name: str,
        message: Optional[str] = None
    ) -> bool:
        """
        Send an invitation email.

        Args:
            to_email: Email of person being invited
            token: Invitation token
            inviter_name: Name of person sending invitation
            entity_type: Type of entity (organization, project, dataset)
            entity_name: Name of the entity
            message: Optional personal message from inviter

        Returns:
            True if email sent successfully
        """
        invite_url = f"{self.frontend_url}/invite/{token}"

        subject = f"You've been invited to join {entity_name}"

        # HTML email body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4F46E5; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: #f9f9f9; padding: 30px; }}
                .button {{ display: inline-block; padding: 12px 24px; background-color: #4F46E5;
                          color: white; text-decoration: none; border-radius: 4px; margin-top: 20px; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Vision AI Training Platform</h1>
                </div>
                <div class="content">
                    <h2>You've been invited!</h2>
                    <p><strong>{inviter_name}</strong> has invited you to join the {entity_type}
                       "<strong>{entity_name}</strong>" on Vision AI Training Platform.</p>

                    {f'<p><em>Personal message:</em> {message}</p>' if message else ''}

                    <p>Click the button below to accept the invitation and create your account:</p>

                    <a href="{invite_url}" class="button">Accept Invitation</a>

                    <p style="margin-top: 20px; font-size: 14px;">
                        Or copy and paste this URL into your browser:<br>
                        <a href="{invite_url}">{invite_url}</a>
                    </p>

                    <p style="margin-top: 30px; color: #666;">
                        This invitation will expire in 7 days.
                    </p>
                </div>
                <div class="footer">
                    <p>Vision AI Training Platform - Natural Language Model Training</p>
                    <p>If you did not expect this invitation, you can safely ignore this email.</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Plain text fallback
        text_body = f"""
        You've been invited to join Vision AI Training Platform!

        {inviter_name} has invited you to join the {entity_type} "{entity_name}".

        {f'Personal message: {message}' if message else ''}

        To accept the invitation, visit:
        {invite_url}

        This invitation will expire in 7 days.

        ---
        Vision AI Training Platform
        """

        return self._send_email(to_email, subject, html_body, text_body)

    def send_verification_email(
        self,
        to_email: str,
        verification_token: str,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Send an email verification email.

        Args:
            to_email: User's email address
            verification_token: Email verification token
            user_name: User's name (optional)

        Returns:
            True if email sent successfully
        """
        verify_url = f"{self.frontend_url}/verify-email/{verification_token}"

        greeting = f"Hi {user_name}," if user_name else "Hi,"

        subject = "Verify your email address"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4F46E5; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: #f9f9f9; padding: 30px; }}
                .button {{ display: inline-block; padding: 12px 24px; background-color: #4F46E5;
                          color: white; text-decoration: none; border-radius: 4px; margin-top: 20px; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Vision AI Training Platform</h1>
                </div>
                <div class="content">
                    <h2>Verify your email address</h2>
                    <p>{greeting}</p>
                    <p>Thank you for signing up for Vision AI Training Platform.
                       Please verify your email address by clicking the button below:</p>

                    <a href="{verify_url}" class="button">Verify Email</a>

                    <p style="margin-top: 20px; font-size: 14px;">
                        Or copy and paste this URL into your browser:<br>
                        <a href="{verify_url}">{verify_url}</a>
                    </p>

                    <p style="margin-top: 30px; color: #666;">
                        This link will expire in 24 hours.
                    </p>
                </div>
                <div class="footer">
                    <p>Vision AI Training Platform</p>
                    <p>If you did not create an account, you can safely ignore this email.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_body = f"""
        Verify your email address

        {greeting}

        Thank you for signing up for Vision AI Training Platform.
        Please verify your email address by visiting:

        {verify_url}

        This link will expire in 24 hours.

        ---
        Vision AI Training Platform
        """

        return self._send_email(to_email, subject, html_body, text_body)

    def send_password_reset_email(
        self,
        to_email: str,
        reset_token: str,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Send a password reset email.

        Args:
            to_email: User's email address
            reset_token: Password reset token
            user_name: User's name (optional)

        Returns:
            True if email sent successfully
        """
        reset_url = f"{self.frontend_url}/reset-password/{reset_token}"

        greeting = f"Hi {user_name}," if user_name else "Hi,"

        subject = "Reset your password"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4F46E5; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: #f9f9f9; padding: 30px; }}
                .button {{ display: inline-block; padding: 12px 24px; background-color: #4F46E5;
                          color: white; text-decoration: none; border-radius: 4px; margin-top: 20px; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
                .warning {{ background-color: #FFF3CD; border: 1px solid #FFEAA7; padding: 10px;
                           border-radius: 4px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Vision AI Training Platform</h1>
                </div>
                <div class="content">
                    <h2>Reset your password</h2>
                    <p>{greeting}</p>
                    <p>We received a request to reset your password.
                       Click the button below to choose a new password:</p>

                    <a href="{reset_url}" class="button">Reset Password</a>

                    <p style="margin-top: 20px; font-size: 14px;">
                        Or copy and paste this URL into your browser:<br>
                        <a href="{reset_url}">{reset_url}</a>
                    </p>

                    <div class="warning">
                        <strong>Security Note:</strong> This link will expire in 1 hour.
                        If you did not request a password reset, please ignore this email.
                    </div>
                </div>
                <div class="footer">
                    <p>Vision AI Training Platform</p>
                    <p>If you need help, please contact support.</p>
                </div>
            </div>
        </body>
        </html>
        """

        text_body = f"""
        Reset your password

        {greeting}

        We received a request to reset your password.
        To choose a new password, visit:

        {reset_url}

        This link will expire in 1 hour.

        If you did not request a password reset, please ignore this email.

        ---
        Vision AI Training Platform
        """

        return self._send_email(to_email, subject, html_body, text_body)


# Global email service instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get or create the global email service instance."""
    global _email_service

    if _email_service is None:
        _email_service = EmailService()

    return _email_service
