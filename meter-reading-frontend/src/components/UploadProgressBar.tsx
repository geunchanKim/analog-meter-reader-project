/**
 * src/components/UploadProgressBar.tsx
 *
 * Visualizes the pipeline's 3 real stages (Stage 1+2 detection+keypoints,
 * Stage 3 OCR, done) as a 3-segment progress bar -- each segment is worth
 * 33.33%, filled once that stage's status checkpoint is reached.
 *
 * Job status only gives us checkpoints (queued -> stage3_queued -> done),
 * not fine-grained progress within a stage, so the currently-active
 * segment gets a moving striped animation instead of a smoothly
 * incrementing fill -- it communicates "still working on this step"
 * without pretending to know more than the backend actually reports.
 */
import "./UploadProgressBar.css";

interface UploadProgressBarProps {
  status: string;
}

const STEPS = [
  { key: "detect", label: "Detect & locate" },
  { key: "read", label: "Read the scale" },
  { key: "done", label: "Done" },
];

function getProgress(status: string): { percent: number; failed: boolean; activeIndex: number } {
  switch (status) {
    case "queued":
      return { percent: 33.33, failed: false, activeIndex: 0 };
    case "stage3_queued":
      return { percent: 66.66, failed: false, activeIndex: 1 };
    case "done":
      return { percent: 100, failed: false, activeIndex: 2 };
    case "fail":
    case "retake":
      // Both failure modes happen at or before the Stage 1+2 -> Stage 3
      // handoff in the vast majority of cases (detection failure,
      // keypoint failure, QC gate rejection) -- freezing at step 1 is a
      // reasonable approximation given job.status doesn't report exactly
      // which stage failed.
      return { percent: 33.33, failed: true, activeIndex: 0 };
    default:
      return { percent: 0, failed: false, activeIndex: -1 };
  }
}

export function UploadProgressBar({ status }: UploadProgressBarProps) {
  const { percent, failed, activeIndex } = getProgress(status);
  const isDone = status === "done";
  const isActive = !isDone && !failed;

  return (
    <div className="upload-progress">
      <div className="upload-progress__track">
        <div
          className={[
            "upload-progress__fill",
            isActive && "upload-progress__fill--active",
            failed && "upload-progress__fill--failed",
            isDone && "upload-progress__fill--done",
          ].filter(Boolean).join(" ")}
          style={{ width: `${percent}%` }}
          role="progressbar"
          aria-valuenow={Math.round(percent)}
          aria-valuemin={0}
          aria-valuemax={100}
        />
        <span className="upload-progress__marker" style={{ left: "33.33%" }} aria-hidden />
        <span className="upload-progress__marker" style={{ left: "66.66%" }} aria-hidden />
      </div>

      <div className="upload-progress__labels">
        {STEPS.map((step, i) => (
          <span
            key={step.key}
            className={`upload-progress__label ${i <= activeIndex ? "upload-progress__label--reached" : ""}`}
          >
            {step.label}
          </span>
        ))}
      </div>
    </div>
  );
}