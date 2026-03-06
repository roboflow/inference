from lidra.init.environment import init_python_path
from lidra.init.patch import patch_all

init_python_path()
patch_all()

import lidra.init.resolvers
