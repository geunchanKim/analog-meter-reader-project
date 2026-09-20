/**
 * src/hooks/useJobPolling.ts
 *
 * Polls GET /jobs/{jobId} on an interval until the job reaches a
 * terminal state (done / fail / retake), then stops automatically.
 *
 * Pulled out into its own hook rather than written inline inside a
 * screen component, so "poll every N seconds, stop once finished, clean
 * up the timer if the component unmounts early" exists in exactly one
 * place. Any screen that needs to watch a job's progress uses this the
 * same way instead of re-implementing setInterval/cleanup logic itself.
 */
import { useEffect, useRef, useState } from "react";
import { getJobStatus, type JobStatusResponse } from "../api/client";

const POLL_INTERVAL_MS = 2000;

// Anything not in this set means "still working" -- keep polling.
// See db/models.py's Job.status comment on the backend for the full
// list of values that can show up here.
const TERMINAL_STATUSES = new Set(["done", "fail", "retake"]);

interface UseJobPollingResult {
  job: JobStatusResponse | null;
  error: string | null;
  isPolling: boolean;
}

export function useJobPolling(jobId: string | null): UseJobPollingResult {
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  // A ref (not state) for the pending timer handle -- we need to read/
  // clear it from the effect's cleanup function without that cleanup
  // re-running every time the timer changes, which state would cause.
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (jobId === null) {
      setJob(null);
      setError(null);
      setIsPolling(false);
      return;
    }

    // Guards against a late response landing AFTER this effect has
    // already been cleaned up (e.g. jobId changed, or the component
    // unmounted) -- without this, a stale poll could call setState on
    // a hook instance nothing is watching anymore.
    let cancelled = false;

    setIsPolling(true);
    setError(null);

    async function poll() {
      try {
        const result = await getJobStatus(jobId as string);
        if (cancelled) return;

        setJob(result);

        if (TERMINAL_STATUSES.has(result.status)) {
          setIsPolling(false);
          return; // reached done/fail/retake -- stop scheduling further polls
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unknown error");
        setIsPolling(false);
        return;
      }

      timeoutRef.current = setTimeout(poll, POLL_INTERVAL_MS);
    }

    poll();

    return () => {
      cancelled = true;
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [jobId]);

  return { job, error, isPolling };
}