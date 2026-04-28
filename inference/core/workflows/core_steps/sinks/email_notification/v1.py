@contextmanager
def establish_smtp_connection(smtp_server, smtp_port) -> Generator[smtplib.SMTP_SSL, None, None]:
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:  # ❌ always SSL
        yield server