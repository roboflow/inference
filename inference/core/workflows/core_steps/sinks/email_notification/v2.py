# Same pattern, same bug, copy-pasted inside _send_email_using_smtp_server_v2
with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:  # ❌ always SSL