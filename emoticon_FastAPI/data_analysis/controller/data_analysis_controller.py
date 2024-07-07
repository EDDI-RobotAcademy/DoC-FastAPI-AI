from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse

from data_analysis.controller.response_form.data_analysis_response_form import AnalysisResponseForm
from data_analysis.service.analysis_service_impl import AnalysisServiceImpl

analysisRouter = APIRouter()

async def injectAnalysisService() -> AnalysisServiceImpl:
    return AnalysisServiceImpl()
@analysisRouter.get("/lgbm-train")
async def ordersTrain(analysisService: AnalysisServiceImpl =
                      Depends(injectAnalysisService)):

    print(f"lgbm Train()")
    try:
        result = await analysisService.trainModel()
        print(result)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@analysisRouter.post("/lgbm-predict", response_model=AnalysisResponseForm)
async def AnalysisData(age: int, gender: str, analysisService: AnalysisServiceImpl =
Depends(injectAnalysisService)):
    # 애플리케이션 상태에서 모델 가져오기

    # 나이대를 문자열로 변환
    if age < 20:
        age_group = '10대'
    elif age < 30:
        age_group = '20대'
    elif age < 40:
        age_group = '30대'
    elif age < 50:
        age_group = '40대'
    elif age < 60:
        age_group = '50대'
    else:
        age_group = '60대 이상'

    # 예측 수행
    predicted_products = await analysisService.predict(age_group, gender)
    return AnalysisResponseForm(predicted_products=predicted_products)
