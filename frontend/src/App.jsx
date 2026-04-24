import { useMemo, useState } from "react";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [confidence, setConfidence] = useState(0.25);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const colorByClass = useMemo(() => {
    return {
      person: "#f97316",
      bottle: "#10b981",
      cup: "#06b6d4",
      default: "#f43f5e"
    };
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(null);
    setError("");

    if (!file) {
      setPreviewUrl("");
      return;
    }

    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return nextUrl;
    });
  };

  const runDetection = async () => {
    if (!selectedFile) {
      setError("Please choose an image before running detection.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      const response = await fetch(`/api/detect?conf=${confidence.toFixed(2)}`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Detection request failed");
      }

      setResult(data);
    } catch (requestError) {
      setError(requestError.message || "Unable to process image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page">
      <section className="panel">
        <h1>YOLOv8 Product Detection</h1>
        <p>
          Upload a sample image and run real-time object detection using a FastAPI
          backend powered by YOLOv8.
        </p>

        <div className="controls">
          <label className="file-input" htmlFor="imageInput">
            Choose Image
          </label>
          <input id="imageInput" type="file" accept="image/*" onChange={handleFileChange} />

          <label htmlFor="confidenceRange">Confidence: {confidence.toFixed(2)}</label>
          <input
            id="confidenceRange"
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={confidence}
            onChange={(event) => setConfidence(Number(event.target.value))}
          />

          <button type="button" onClick={runDetection} disabled={loading}>
            {loading ? "Detecting..." : "Run Detection"}
          </button>
        </div>

        {error && <p className="error">{error}</p>}
      </section>

      {previewUrl && (
        <section className="results">
          <div className="image-stage">
            <img src={previewUrl} alt="Selected preview" className="preview" />
            {result?.detections?.map((detection, index) => {
              const width = result.width || 1;
              const height = result.height || 1;
              const left = (detection.bbox.x1 / width) * 100;
              const top = (detection.bbox.y1 / height) * 100;
              const boxWidth = (detection.bbox.width / width) * 100;
              const boxHeight = (detection.bbox.height / height) * 100;
              const boxColor = colorByClass[detection.className] || colorByClass.default;

              return (
                <div
                  key={`${detection.className}-${index}`}
                  className="bbox"
                  style={{
                    left: `${left}%`,
                    top: `${top}%`,
                    width: `${boxWidth}%`,
                    height: `${boxHeight}%`,
                    borderColor: boxColor
                  }}
                >
                  <span style={{ background: boxColor }}>
                    {detection.className} ({detection.confidence})
                  </span>
                </div>
              );
            })}
          </div>

          <div className="summary">
            <h2>Detections</h2>
            <p>Total: {result?.detections?.length ?? 0}</p>
            <ul>
              {(result?.detections ?? []).map((item, index) => (
                <li key={`${item.className}-${index}`}>
                  {item.className} | Confidence {item.confidence}
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}
    </main>
  );
}

export default App;
