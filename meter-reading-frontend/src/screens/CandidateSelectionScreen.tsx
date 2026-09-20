/**
 * src/screens/CandidateSelectionScreen.tsx
 *
 * Second screen: shows Stage 3's top-3 candidates for both positions,
 * lets the person pick one per side and swap which side is min/max if
 * the model guessed backwards.
 *
 * Flow is three steps, not two:
 *   1. Pick candidates (+ swap).
 *   2. "Calculate" -- previews the computed reading WITHOUT saving
 *      anything. A person can't judge "correct" or "incorrect" without
 *      first seeing what the calculated value actually is.
 *   3. Judge it -- Correct/Unclear submit as-is; Incorrect asks what the
 *      gauge actually reads first, then submits both numbers.
 *
 * Changing any candidate/swap choice after step 2 resets the preview,
 * forcing a recalculation -- otherwise a judgment could end up attached
 * to a stale number that no longer matches the current selection.
 */
import { useEffect, useState } from "react";
import {
  getGaugeResult,
  previewSelection,
  submitSelection,
  explainCandidates,
  ApiError,
  type CandidateOut,
  type GaugeResultResponse,
  type ExplainResponse,
} from "../api/client";
import { ImageKeypointOverlay } from "../components/ImageKeypointOverlay";
import "./CandidateSelectionScreen.css";

interface CandidateSelectionScreenProps {
  gaugeId: string;
  imageUrl: string;
  onFinished: () => void;
}

type Judgment = "correct" | "incorrect" | "unclear";

export function CandidateSelectionScreen({ gaugeId, imageUrl, onFinished }: CandidateSelectionScreenProps) {
  const [result, setResult] = useState<GaugeResultResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // "custom" means the person typed their own value instead of picking
  // one of the top-3 -- added after a real test photo (the WIKA 0-60°C
  // gauge) came back with "60" missing from ALL THREE max candidates.
  const [selectedA, setSelectedA] = useState<number | "custom">(0);
  const [customA, setCustomA] = useState("");
  const [selectedB, setSelectedB] = useState<number | "custom">(0);
  const [customB, setCustomB] = useState("");
  const [swapped, setSwapped] = useState(false);

  // Step 2: preview, before anything is saved.
  const [previewValue, setPreviewValue] = useState<number | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // Step 3: judging + (for "incorrect") the actual value.
  const [submitting, setSubmitting] = useState<Judgment | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [askingActualValue, setAskingActualValue] = useState(false);
  const [actualValueInput, setActualValueInput] = useState("");

  // Final, saved result.
  const [computedValue, setComputedValue] = useState<number | null>(null);
  const [finalRange, setFinalRange] = useState<{ min: number; max: number } | null>(null);
  const [finalActualValue, setFinalActualValue] = useState<number | null>(null);
  const [finalJudgment, setFinalJudgment] = useState<Judgment | null>(null);

  // AI agent -- advisory only, fetched on demand rather than automatically,
  // since it costs an LLM call and isn't needed for the core flow to work.
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [explaining, setExplaining] = useState(false);

  useEffect(() => {
    getGaugeResult(gaugeId)
      .then(setResult)
      .catch((err) => setLoadError(err instanceof ApiError ? err.message : "Couldn't load candidates"));
  }, [gaugeId]);

  // Any change to the selection invalidates a previous preview -- forces
  // "Calculate" again before judging, so the two can't drift apart.
  useEffect(() => {
    setPreviewValue(null);
    setAskingActualValue(false);
  }, [selectedA, customA, selectedB, customB, swapped]);

  if (loadError) {
    return (
      <div className="candidate-screen">
        <ImageKeypointOverlay imageUrl={imageUrl} />
        <p className="candidate-screen__error">{loadError}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="candidate-screen">
        <ImageKeypointOverlay imageUrl={imageUrl} />
        <p className="candidate-screen__loading">Loading candidates…</p>
      </div>
    );
  }

  const valueA = resolveValue(selectedA, result.pos_a_candidates, customA);
  const valueB = resolveValue(selectedB, result.pos_b_candidates, customB);
  const hasValidSelection = valueA !== null && valueB !== null;
  const chosenMin = valueA !== null && valueB !== null ? (swapped ? valueB : valueA) : null;
  const chosenMax = valueA !== null && valueB !== null ? (swapped ? valueA : valueB) : null;

  async function handleCalculate() {
    if (chosenMin === null || chosenMax === null) return;
    setPreviewing(true);
    setPreviewError(null);
    try {
      const res = await previewSelection(gaugeId, { chosen_min_value: chosenMin, chosen_max_value: chosenMax, swapped });
      setPreviewValue(res.computed_value);
    } catch (err) {
      setPreviewError(err instanceof ApiError ? err.message : "Couldn't calculate a reading");
    } finally {
      setPreviewing(false);
    }
  }

  async function handleAskAI() {
    setExplaining(true);
    try {
      const res = await explainCandidates(gaugeId);
      setExplanation(res);
    } catch {
      // Advisory only -- a failed explanation should never block picking
      // candidates or judging the reading, so this fails silently rather
      // than surfacing an error banner for a non-essential feature.
    } finally {
      setExplaining(false);
    }
  }

  async function handleJudgment(judgment: Judgment, actualValue?: number) {
    if (chosenMin === null || chosenMax === null) return;
    setSubmitting(judgment);
    setSubmitError(null);

    try {
      const res = await submitSelection(gaugeId, {
        chosen_min_value: chosenMin,
        chosen_max_value: chosenMax,
        swapped,
        judgment,
        actual_value: actualValue,
      });
      setComputedValue(res.computed_value);
      setFinalRange({ min: chosenMin, max: chosenMax });
      setFinalActualValue(actualValue ?? null);
      setFinalJudgment(judgment);
    } catch (err) {
      setSubmitError(err instanceof ApiError ? err.message : "Couldn't save your selection");
    } finally {
      setSubmitting(null);
    }
  }

  if (computedValue !== null) {
    return (
      <div className="candidate-screen">
        <ImageKeypointOverlay imageUrl={imageUrl} keypoints={result.keypoints} cropBbox={result.bbox} />
        <p className="candidate-screen__result-label">Reading</p>
        <p className={`candidate-screen__result-value candidate-screen__result-value--${finalJudgment ?? "unclear"}`}>
          {computedValue.toFixed(2)}
        </p>
        {finalRange && (
          <p className="candidate-screen__result-range">
            Min {finalRange.min} &middot; Max {finalRange.max}
          </p>
        )}
        {finalActualValue !== null && (
          <p className="candidate-screen__result-actual">
            You said it actually reads {finalActualValue}
          </p>
        )}
        <button className="candidate-screen__reset" onClick={onFinished}>
          Read another gauge
        </button>
      </div>
    );
  }

  return (
    <div className="candidate-screen">
      <header className="candidate-screen__header">
        <h2 className="candidate-screen__title">Confirm the scale</h2>
        <p className="candidate-screen__subtitle">
          Pick the correct min and max label. Swap if the model guessed backwards.
        </p>
      </header>

      <ImageKeypointOverlay imageUrl={imageUrl} keypoints={result.keypoints} cropBbox={result.bbox} />

      <button className="candidate-screen__ask-ai" onClick={handleAskAI} disabled={explaining}>
        {explaining ? "Thinking…" : explanation ? "Ask AI again" : "Ask AI"}
      </button>

      <div className="candidate-screen__columns">
        <CandidateColumn
          label={swapped ? "Max" : "Min"}
          candidates={result.pos_a_candidates}
          selected={selectedA}
          onSelect={setSelectedA}
          customValue={customA}
          note={explanation?.position_a ?? undefined}
          onCustomValueChange={(v) => {
            setCustomA(v);
            setSelectedA("custom");
          }}
        />

        <button
          className="candidate-screen__swap"
          onClick={() => setSwapped((s) => !s)}
          aria-label="Swap min and max"
          title="Swap min and max"
        >
          ⇄
        </button>

        <CandidateColumn
          label={swapped ? "Min" : "Max"}
          candidates={result.pos_b_candidates}
          selected={selectedB}
          onSelect={setSelectedB}
          customValue={customB}
          note={explanation?.position_b ?? undefined}
          onCustomValueChange={(v) => {
            setCustomB(v);
            setSelectedB("custom");
          }}
        />
      </div>

      {previewError && <p className="candidate-screen__error">{previewError}</p>}
      {submitError && <p className="candidate-screen__error">{submitError}</p>}

      {previewValue === null ? (
        <button
          className="candidate-screen__calculate"
          disabled={!hasValidSelection || previewing}
          onClick={handleCalculate}
        >
          {previewing ? "Calculating…" : "Calculate reading"}
        </button>
      ) : (
        <>
          <div className="candidate-screen__preview">
            <span className="candidate-screen__preview-label">Calculated reading</span>
            <span className="candidate-screen__preview-value">{previewValue.toFixed(2)}</span>
          </div>

          {askingActualValue ? (
            <div className="candidate-screen__actual-value">
              <label className="candidate-screen__actual-value-label" htmlFor="actual-value-input">
                What does it actually read?
              </label>
              <div className="candidate-screen__actual-value-row">
                <input
                  id="actual-value-input"
                  type="text"
                  inputMode="decimal"
                  autoFocus
                  placeholder="e.g. 42.5"
                  className="candidate-screen__actual-value-input"
                  value={actualValueInput}
                  onChange={(e) => setActualValueInput(e.target.value)}
                />
                <button
                  className="candidate-screen__actual-value-submit"
                  disabled={submitting !== null || Number.isNaN(Number(actualValueInput)) || actualValueInput.trim() === ""}
                  onClick={() => handleJudgment("incorrect", Number(actualValueInput))}
                >
                  Submit
                </button>
              </div>
              <button
                className="candidate-screen__actual-value-cancel"
                onClick={() => {
                  setAskingActualValue(false);
                  setActualValueInput("");
                }}
              >
                Cancel
              </button>
            </div>
          ) : (
            <div className="candidate-screen__judgments">
              <button
                className="candidate-screen__judgment candidate-screen__judgment--correct"
                disabled={submitting !== null}
                onClick={() => handleJudgment("correct")}
              >
                Correct
              </button>
              <button
                className="candidate-screen__judgment candidate-screen__judgment--incorrect"
                disabled={submitting !== null}
                onClick={() => setAskingActualValue(true)}
              >
                Incorrect
              </button>
              <button
                className="candidate-screen__judgment candidate-screen__judgment--unclear"
                disabled={submitting !== null}
                onClick={() => handleJudgment("unclear")}
              >
                Unclear
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** Resolves the numeric value a column currently represents -- either
 * the parsed_value of the selected candidate, or a manually-typed
 * value, parsed as a float. Returns null while the choice is invalid/
 * incomplete, which the "Calculate" button uses to stay disabled. */
function resolveValue(
  selected: number | "custom",
  candidates: CandidateOut[],
  customValue: string,
): number | null {
  if (selected === "custom") {
    if (customValue.trim() === "") return null;
    const parsed = Number(customValue);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return candidates[selected]?.parsed_value ?? null;
}

interface CandidateColumnProps {
  label: string;
  candidates: CandidateOut[];
  selected: number | "custom";
  onSelect: (index: number | "custom") => void;
  customValue: string;
  onCustomValueChange: (value: string) => void;
  note?: string;
}

function CandidateColumn({
  label,
  candidates,
  selected,
  onSelect,
  customValue,
  onCustomValueChange,
  note,
}: CandidateColumnProps) {
  return (
    <div className="candidate-column">
      <span className="candidate-column__label">{label}</span>
      {candidates.map((c, i) => (
        <label key={i} className="candidate-option">
          <input
            type="radio"
            name={label}
            checked={selected === i}
            onChange={() => onSelect(i)}
          />
          <span className="candidate-option__value">{c.text}</span>
        </label>
      ))}

      {/* Manual override -- typing in this field selects it automatically,
          no separate click needed. Exists specifically for the case where
          none of the top-3 candidates are the real value. */}
      <label className="candidate-option candidate-option--custom">
        <input
          type="radio"
          name={label}
          checked={selected === "custom"}
          onChange={() => onSelect("custom")}
        />
        <input
          type="text"
          inputMode="decimal"
          placeholder="Other…"
          className="candidate-option__input"
          value={customValue}
          onChange={(e) => onCustomValueChange(e.target.value)}
        />
      </label>

      {note && <p className="candidate-column__note">{note}</p>}
    </div>
  );
}