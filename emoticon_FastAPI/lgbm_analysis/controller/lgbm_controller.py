import joblib
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from emoticon_FastAPI.lgbm_analysis.controller.request_form.predict_request_form import PredictRequestForm
from emoticon_FastAPI.lgbm_analysis.service.lgbm_service_impl import LgbmAnalysisServiceImpl

lgbmAnalysisRouter = APIRouter()


async def injectlgbmAnalysisService() -> LgbmAnalysisServiceImpl:
    return LgbmAnalysisServiceImpl()


@lgbmAnalysisRouter.get("/lgbm-train")
async def lgbmTrain(lgbmAnalysisService: LgbmAnalysisServiceImpl =
                      Depends(injectlgbmAnalysisService)):

    print(f"controller -> lgbmTrain()")

    try:
        result = await lgbmAnalysisService.lgbmTrain()
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
