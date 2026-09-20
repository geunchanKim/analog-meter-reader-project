/**
 * src/screens/UploadScreen.tsx
 *
 * First screen: upload a photo, watch it move through the pipeline.
 * The uploaded photo now stays visible the whole time (via an object
 * URL created client-side) instead of disappearing right after the
 * dropzone interaction -- and once the job is "done", that same URL is
 * handed up to the parent so later screens can keep showing it.
 */
import { useEffect, useRef, useState } from "react";
import { PhotoDropzone } from "../components/PhotoDropzone";
import { ImageKeypointOverlay } from "../components/ImageKeypointOverlay";
import { UploadProgressBar } from "../components/UploadProgressBar";
import { enqueueGauge, ApiError } from "../api/client";
import { useJobPolling } from "../hooks/useJobPolling";
import "./UploadScreen.css";

interface UploadScreenProps {
  onGaugeReady: (gaugeId: string, imageUrl: string) => void;
}

const STATUS_COPY: Record<string, string> = {
  queued: "Queued",
  stage3_queued: "Reading the scale",
  fail: "Couldn't read this photo",
  retake: "Needs a retake",
  done: "Done",
};

export function UploadScreen({ onGaugeReady }: UploadScreenProps) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const { job, error: pollError, isPolling } = useJobPolling(jobId);

  // NOTE: deliberately NOT revoking this URL on unmount. Once
  // onGaugeReady hands it to App.tsx, CandidateSelectionScreen keeps
  // using the SAME url after this component unmounts -- revoking it
  // here (which an unmount-cleanup effect originally did) killed the
  // image on the very next screen. It's only revoked when THIS screen
  // itself creates a replacement (a new file picked), see handleFileSelected.
  const imageUrlRef = useRef<string | null>(null);

  async function handleFileSelected(file: File) {
    setUploadError(null);

    if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    const url = URL.createObjectURL(file);
    imageUrlRef.current = url;
    setImageUrl(url);

    try {
      const { job_id } = await enqueueGauge(file);
      setJobId(job_id);
    } catch (err) {
      setUploadError(err instanceof ApiError ? err.message : "Upload failed");
    }
  }

  function handleRetry() {
    setJobId(null);
    setUploadError(null);
    if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    imageUrlRef.current = null;
    setImageUrl(null);
  }

  const isBusy = jobId !== null && job?.status !== "fail" && job?.status !== "retake";

  useEffect(() => {
    if (job?.status === "done" && job.gauge_id && imageUrl) {
      onGaugeReady(job.gauge_id, imageUrl);
    }
  }, [job?.status, job?.gauge_id, imageUrl, onGaugeReady]);

  return (
    <div className="upload-screen">
      <header className="upload-screen__header">
        <h1 className="upload-screen__title">Meter Reader</h1>
        <p className="upload-screen__subtitle">
          Photograph an analog gauge and read the value off its needle.
        </p>
      </header>

      {imageUrl ? (
        <ImageKeypointOverlay imageUrl={imageUrl} />
      ) : (
        <PhotoDropzone onFileSelected={handleFileSelected} disabled={isBusy} />
      )}

      {uploadError && <p className="upload-screen__error">{uploadError}</p>}

      {job && <UploadProgressBar status={job.status} />}

      {job && (
        <div className="upload-screen__status">
          <div className="upload-screen__status-row">
            <span className="upload-screen__status-label">Status</span>
            <span
              className={`upload-screen__status-value upload-screen__status-value--${job.status}`}
            >
              {STATUS_COPY[job.status] ?? job.status}
              {isPolling && <span className="upload-screen__pulse" aria-hidden />}
            </span>
          </div>

          {job.detail && (job.status === "fail" || job.status === "retake") && (
            <p className="upload-screen__detail">{job.detail}</p>
          )}
        </div>
      )}

      {(job?.status === "fail" || job?.status === "retake") && (
        <button className="upload-screen__retry" onClick={handleRetry}>
          Try another photo
        </button>
      )}

      {pollError && <p className="upload-screen__error">{pollError}</p>}
    </div>
  );
}