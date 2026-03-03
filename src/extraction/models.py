from enum import Enum
from pydantic import BaseModel


class EntityType(str, Enum):
    KHAI_NIEM = "khái_niệm"        # Concept/Definition
    DIEU_LUAT = "điều_luật"        # Article/Legal provision
    QUYEN_NGHIA_VU = "quyền_nghĩa_vụ"  # Rights/Obligations
    MUC_HUONG = "mức_hưởng"        # Benefits/Entitlements
    XU_PHAT = "xử_phạt"           # Penalties/Sanctions


class RelationType(str, Enum):
    DINH_NGHIA = "định_nghĩa"          # defines
    QUY_DINH = "quy_định"              # regulates
    AP_DUNG = "áp_dụng"                # applies to
    THAM_CHIEU = "tham_chiếu"          # references
    BAO_GOM = "bao_gồm"               # includes
    DIEU_KIEN = "điều_kiện"            # condition for
    HAN_CHE = "hạn_chế"               # restricts
    LIEN_QUAN = "liên_quan"            # related to


class Entity(BaseModel):
    id: str
    name: str
    entity_type: EntityType
    description: str
    source_article: int
    source_text: str = ""
    law_id: str = "2024"

    @property
    def embedding_text(self) -> str:
        return f"{self.name} (LĐĐ {self.law_id}): {self.description}"


class Relation(BaseModel):
    source_id: str
    target_id: str
    relation_type: RelationType
    description: str
    source_article: int
    law_id: str = "2024"


class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
