from .d4 import D4_codebook
from .e8p12 import E8P12_codebook
from .hi import HI4B1C_codebook
from .e8p12_rvq3 import E8P12RVQ3B_codebook
from .e8p12_rvq4 import E8P12RVQ4B_codebook

codebook_id = {
    "D4": D4_codebook,
    "E8P12": E8P12_codebook,
    "HI": HI4B1C_codebook,
    "E8P12RVQ3B": E8P12RVQ3B_codebook,
    "E8P12RVQ4B": E8P12RVQ4B_codebook
}
