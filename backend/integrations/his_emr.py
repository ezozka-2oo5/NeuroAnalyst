"""
backend/integrations/his_emr.py
=================================
HIS/EMR (Hospital Information System / Electronic Medical Records) integratsiyasi.

Qo'llab-quvvatlanadigan protokollar:
  - HL7 v2.x (ADT, ORM, ORU xabarlari)
  - FHIR R4 (Patient, Observation, DiagnosticReport resurslari)
  - REST API (custom HIS tizimlari uchun)

Funksiyalar:
  - Bemor ma'lumotlarini HIS dan olish
  - EEG natijalarini HIS ga yuborish
  - Real-time xabarlar almashish
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from backend.config import get_settings
from backend.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class HISIntegration:
    """
    HIS/EMR tizimi bilan integratsiya klassi.
    
    Sozlash:
        .env faylida quyidagi parametrlarni belgilang:
        - HIS_ENABLED=true
        - HIS_TYPE=hl7|fhir|rest
        - HIS_BASE_URL=http://his.hospital.local:8080
        - HIS_API_KEY=your_api_key_here
    """
    
    def __init__(self):
        self.enabled = settings.his_enabled
        self.his_type = settings.his_type
        self.base_url = settings.his_base_url
        self.api_key = settings.his_api_key
        self.timeout = 30.0
        
        if self.enabled:
            logger.info(f"HIS integratsiya yoqilgan: {self.his_type} @ {self.base_url}")
        else:
            logger.info("HIS integratsiya o'chirilgan")
    
    async def get_patient_by_mrn(self, mrn: str) -> Optional[Dict[str, Any]]:
        """
        MRN (Medical Record Number) bo'yicha bemor ma'lumotlarini olish.
        
        Args:
            mrn: Bemor tibbiy rekord raqami
            
        Returns:
            Bemor ma'lumotlari dict yoki None
        """
        if not self.enabled:
            logger.warning("HIS integratsiya o'chirilgan")
            return None
        
        try:
            if self.his_type == "fhir":
                return await self._get_patient_fhir(mrn)
            elif self.his_type == "hl7":
                return await self._get_patient_hl7(mrn)
            elif self.his_type == "rest":
                return await self._get_patient_rest(mrn)
            else:
                logger.error(f"Noma'lum HIS turi: {self.his_type}")
                return None
        except Exception as e:
            logger.error(f"HIS dan bemor olishda xato: {e}")
            return None
    
    async def send_eeg_report(
        self,
        patient_mrn: str,
        study_data: Dict[str, Any],
        report_text: str,
        pdf_bytes: Optional[bytes] = None,
    ) -> bool:
        """
        EEG natijalarini HIS ga yuborish.
        
        Args:
            patient_mrn: Bemor MRN
            study_data: EEG tekshiruv ma'lumotlari
            report_text: Hisobot matni
            pdf_bytes: PDF hisobot (agar mavjud bo'lsa)
            
        Returns:
            True - muvaffaqiyatli, False - xato
        """
        if not self.enabled:
            logger.warning("HIS integratsiya o'chirilgan")
            return False
        
        try:
            if self.his_type == "fhir":
                return await self._send_report_fhir(patient_mrn, study_data, report_text, pdf_bytes)
            elif self.his_type == "hl7":
                return await self._send_report_hl7(patient_mrn, study_data, report_text)
            elif self.his_type == "rest":
                return await self._send_report_rest(patient_mrn, study_data, report_text, pdf_bytes)
            else:
                logger.error(f"Noma'lum HIS turi: {self.his_type}")
                return False
        except Exception as e:
            logger.error(f"HIS ga hisobot yuborishda xato: {e}")
            return False
    
    # ── FHIR R4 implementatsiyasi ──────────────────────────────────────
    
    async def _get_patient_fhir(self, mrn: str) -> Optional[Dict[str, Any]]:
        """FHIR Patient resursini olish."""
        url = f"{self.base_url}/Patient"
        params = {"identifier": mrn}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/fhir+json",
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            bundle = response.json()
            if bundle.get("total", 0) == 0:
                logger.warning(f"FHIR: Bemor topilmadi (MRN={mrn})")
                return None
            
            patient = bundle["entry"][0]["resource"]
            return self._parse_fhir_patient(patient)
    
    async def _send_report_fhir(
        self,
        patient_mrn: str,
        study_data: Dict[str, Any],
        report_text: str,
        pdf_bytes: Optional[bytes],
    ) -> bool:
        """FHIR DiagnosticReport yaratish."""
        url = f"{self.base_url}/DiagnosticReport"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/fhir+json",
        }
        
        # FHIR DiagnosticReport resursi
        diagnostic_report = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "EEG",
                    "display": "Electroencephalogram"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "11502-2",
                    "display": "EEG study"
                }]
            },
            "subject": {
                "identifier": {"value": patient_mrn}
            },
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
            "issued": datetime.utcnow().isoformat() + "Z",
            "conclusion": report_text,
            "conclusionCode": [{
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "17621005" if study_data.get("label") == "Normal" else "84757009",
                    "display": study_data.get("label", "Unknown")
                }]
            }]
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=diagnostic_report, headers=headers)
            response.raise_for_status()
            logger.info(f"FHIR: DiagnosticReport yuborildi (MRN={patient_mrn})")
            return True
    
    def _parse_fhir_patient(self, fhir_patient: Dict) -> Dict[str, Any]:
        """FHIR Patient resursini ichki formatga o'tkazish."""
        name = fhir_patient.get("name", [{}])[0]
        return {
            "patient_code": fhir_patient.get("id"),
            "full_name": f"{name.get('given', [''])[0]} {name.get('family', '')}".strip(),
            "gender": fhir_patient.get("gender", "unknown"),
            "birth_date": fhir_patient.get("birthDate"),
            "mrn": next(
                (id["value"] for id in fhir_patient.get("identifier", []) if id.get("system") == "MRN"),
                None
            ),
        }
    
    # ── HL7 v2.x implementatsiyasi ─────────────────────────────────────
    
    async def _get_patient_hl7(self, mrn: str) -> Optional[Dict[str, Any]]:
        """HL7 QRY^A19 xabari orqali bemor so'rash."""
        # HL7 xabar yaratish
        hl7_message = self._build_hl7_query(mrn)
        
        url = f"{self.base_url}/hl7"
        headers = {"Content-Type": "text/plain"}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, content=hl7_message, headers=headers)
            response.raise_for_status()
            
            # HL7 javobni parse qilish
            return self._parse_hl7_response(response.text)
    
    async def _send_report_hl7(
        self,
        patient_mrn: str,
        study_data: Dict[str, Any],
        report_text: str,
    ) -> bool:
        """HL7 ORU^R01 xabari orqali natija yuborish."""
        hl7_message = self._build_hl7_result(patient_mrn, study_data, report_text)
        
        url = f"{self.base_url}/hl7"
        headers = {"Content-Type": "text/plain"}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, content=hl7_message, headers=headers)
            response.raise_for_status()
            logger.info(f"HL7: Natija yuborildi (MRN={patient_mrn})")
            return True
    
    def _build_hl7_query(self, mrn: str) -> str:
        """HL7 QRY^A19 xabari yaratish."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return (
            f"MSH|^~\\&|NeuroAnalyst|CLINIC|HIS|HOSPITAL|{timestamp}||QRY^A19|{timestamp}|P|2.5\r"
            f"QRD|{timestamp}|R|I|{mrn}|||1^RD|{mrn}|DEM|\r"
        )
    
    def _build_hl7_result(self, mrn: str, study_data: Dict, report_text: str) -> str:
        """HL7 ORU^R01 xabari yaratish."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return (
            f"MSH|^~\\&|NeuroAnalyst|CLINIC|HIS|HOSPITAL|{timestamp}||ORU^R01|{timestamp}|P|2.5\r"
            f"PID|||{mrn}|||||||||||||||||||||||||||\r"
            f"OBR|1||{study_data.get('id', '')}|EEG^Electroencephalogram|||{timestamp}\r"
            f"OBX|1|TX|EEG^Result||{study_data.get('label', 'Unknown')}||||||F\r"
            f"OBX|2|NM|RISK^Seizure Risk||{study_data.get('seizure_risk', 0)}||||||F\r"
            f"NTE|1||{report_text[:200]}\r"
        )
    
    def _parse_hl7_response(self, hl7_text: str) -> Optional[Dict[str, Any]]:
        """HL7 javobni parse qilish (soddalashtirilgan)."""
        lines = hl7_text.split("\r")
        pid_line = next((l for l in lines if l.startswith("PID")), None)
        
        if not pid_line:
            return None
        
        fields = pid_line.split("|")
        return {
            "patient_code": fields[3] if len(fields) > 3 else "",
            "full_name": fields[5] if len(fields) > 5 else "",
            "mrn": fields[3] if len(fields) > 3 else "",
        }
    
    # ── REST API implementatsiyasi ─────────────────────────────────────
    
    async def _get_patient_rest(self, mrn: str) -> Optional[Dict[str, Any]]:
        """REST API orqali bemor olish."""
        url = f"{self.base_url}/api/patients/{mrn}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
    
    async def _send_report_rest(
        self,
        patient_mrn: str,
        study_data: Dict[str, Any],
        report_text: str,
        pdf_bytes: Optional[bytes],
    ) -> bool:
        """REST API orqali hisobot yuborish."""
        url = f"{self.base_url}/api/reports"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "patient_mrn": patient_mrn,
            "report_type": "EEG",
            "report_date": datetime.utcnow().isoformat(),
            "findings": study_data,
            "report_text": report_text,
        }
        
        if pdf_bytes:
            import base64
            payload["pdf_base64"] = base64.b64encode(pdf_bytes).decode("utf-8")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"REST: Hisobot yuborildi (MRN={patient_mrn})")
            return True


# Singleton instance
_his_integration: Optional[HISIntegration] = None


def get_his_integration() -> HISIntegration:
    """HIS integratsiya singleton instance."""
    global _his_integration
    if _his_integration is None:
        _his_integration = HISIntegration()
    return _his_integration

# Made with Bob
