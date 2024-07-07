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

@lgbmAnalysisRouter.post("/lgbm-predict")
async def lgbmPredict(request: PredictRequestForm,
                        lgbmAnalysisService: LgbmAnalysisServiceImpl =
                        Depends(injectlgbmAnalysisService)):

    print(f"controller -> lgbmPredict()")
    predictedCategory = await lgbmAnalysisService.lgbmPredict(age=request.age, gender=request.gender)
    recommendIds = lgbmAnalysisService.getRecommendProducts(category=predictedCategory, k=5)

    return JSONResponse(content={"prediction": predictedCategory, "recommended IDs": recommendIds}, status_code=status.HTTP_200_OK)