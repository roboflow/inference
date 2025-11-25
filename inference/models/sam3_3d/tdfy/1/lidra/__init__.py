import os

# Allow skipping initialization for lightweight tools
if not os.environ.get('LIDRA_SKIP_INIT'):
    import lidra.init
