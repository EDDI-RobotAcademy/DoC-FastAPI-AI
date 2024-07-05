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
async def AnalysisData(AnalysisService: AnalysisServiceImpl =
                             Depends(injectAnalysisService)):

    return await AnalysisService.predict()
