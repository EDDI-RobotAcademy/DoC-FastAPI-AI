from fastapi import APIRouter
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from fastapi.responses import StreamingResponse
import io
import matplotlib.pyplot as plt

from data_analysis.controller.response_form.kmeans_cluster_response_form import KmeansClusterResponseForm

kmeansRouter = APIRouter()


@kmeansRouter.get("/kmeans-test", response_model=KmeansClusterResponseForm)
async def kmeans_cluster_analysis():
    X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)

    kmeans = KMeans(n_clusters=4, n_init=10)
    kmeans.fit(X)

    labels = kmeans.labels_.tolist()
    centers = kmeans.cluster_centers_.tolist()
    points = X.tolist()

    # SSE 계산
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    return {"centers": centers, "labels": labels, "points": points, "sse": sse}


@kmeansRouter.get("/kmeans-scatterplot")
async def kmeans_scatterplot():
    X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=4, n_init=10)
    kmeans.fit(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title("K-Means Clustering Scatter Plot")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")
