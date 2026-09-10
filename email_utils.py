import os
import requests


def send_verification_email(to_email: str, verify_url: str):
    """Send email verification link via Brevo HTTP API."""

    api_key = os.getenv("BREVO_API_KEY")
    smtp_from = os.getenv("SMTP_FROM", "noreply@markremoverai.com")

    print(f"[EMAIL DEBUG] Sending verification email via Brevo")
    print(f"[EMAIL DEBUG] From: {smtp_from}")
    print(f"[EMAIL DEBUG] To: {to_email}")

    if not api_key:
        raise ValueError("Missing BREVO_API_KEY environment variable")

    url = "https://api.brevo.com/v3/smtp/email"

    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }

    payload = {
        "sender": {
            "name": "MarkRemoverAI",
            "email": smtp_from
        },
        "to": [
            {
                "email": to_email
            }
        ],
        "subject": "Verify your MarkRemoverAI email",
        "textContent": (
            f"Welcome to MarkRemoverAI!\n\n"
            f"Click the link below to verify your email address:\n\n"
            f"{verify_url}\n\n"
            f"This link expires in 24 hours.\n\n"
            "If you didn't create an account, please ignore this email."
        ),
        "htmlContent": f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f4f4;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f4f4f4;">
        <tr>
            <td align="center" style="padding: 40px 20px;">
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);">
                    <tr>
                        <td align="center" style="padding: 40px 40px 30px;">
                            <h1 style="margin: 0; color: #667eea; font-size: 28px; font-weight: bold;">MarkRemoverAI</h1>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0 40px;">
                            <h2 style="margin: 0 0 20px; color: #333333; font-size: 22px;">Welcome!</h2>
                            <p style="margin: 0 0 25px; color: #555555; font-size: 16px; line-height: 1.6;">
                                Thank you for signing up. Please verify your email address by clicking the button below:
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td align="center" style="padding: 0 40px 30px;">
                            <a href="{verify_url}" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; text-decoration: none; padding: 14px 40px; border-radius: 8px; font-size: 16px; font-weight: bold;">
                                Verify Email Address
                            </a>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0 40px 30px;">
                            <p style="margin: 0 0 15px; color: #777777; font-size: 14px; line-height: 1.5;">
                                Or copy and paste this link into your browser:
                            </p>
                            <p style="margin: 0; color: #667eea; font-size: 13px; word-break: break-all;">
                                {verify_url}
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0 40px 40px;">
                            <p style="margin: 0; color: #999999; font-size: 13px;">
                                This link expires in 24 hours.<br>
                                If you didn't create an account, please ignore this email.
                            </p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 20px 40px; background-color: #f8f9fa; border-radius: 0 0 12px 12px;">
                            <p style="margin: 0; color: #999999; font-size: 12px; text-align: center;">
                                &copy; 2026 MarkRemoverAI. All rights reserved.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    }

    print(f"[EMAIL DEBUG] Sending verification email...")

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 201:
        print(f"[EMAIL SUCCESS] Verification email sent to {to_email}")
        return True
    else:
        error_msg = f"Brevo API error {response.status_code}: {response.text}"
        print(f"[EMAIL ERROR] {error_msg}")
        raise Exception(error_msg)


def send_reset_email(to_email: str, reset_url: str):
    """Send password reset email via Brevo HTTP API."""

    api_key = os.getenv("BREVO_API_KEY")
    smtp_from = os.getenv("SMTP_FROM", "noreply@markremoverai.com")

    print(f"[EMAIL DEBUG] Using Brevo HTTP API")
    print(f"[EMAIL DEBUG] From: {smtp_from}")
    print(f"[EMAIL DEBUG] To: {to_email}")
    print(f"[EMAIL DEBUG] API Key: {'*' * 20}...{api_key[-8:] if api_key else 'NOT SET'}")

    if not api_key:
        raise ValueError("Missing BREVO_API_KEY environment variable")

    url = "https://api.brevo.com/v3/smtp/email"

    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }

    payload = {
        "sender": {
            "name": "MarkRemoverAI",
            "email": smtp_from
        },
        "to": [
            {
                "email": to_email
            }
        ],
        "subject": "Reset your MarkRemoverAI password",
        "textContent": (
            f"Click the link below to reset your password (expires in 1 hour):\n\n"
            f"{reset_url}\n\n"
            "If you didn't request this, please ignore this email."
        )
    }

    print(f"[EMAIL DEBUG] Sending via Brevo API...")

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 201:
        print(f"[EMAIL SUCCESS] Reset link sent to {to_email}")
        return True
    else:
        error_msg = f"Brevo API error {response.status_code}: {response.text}"
        print(f"[EMAIL ERROR] {error_msg}")
        raise Exception(error_msg)
