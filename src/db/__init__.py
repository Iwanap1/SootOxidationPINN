from dotenv import load_dotenv
load_dotenv()
import warnings
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="You appear to be connected to a CosmosDB cluster.*",
    category=UserWarning,
)

from .db import DB