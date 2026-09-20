/**
 * src/components/HistorySidebar.tsx
 *
 * Persistent left sidebar (like a chat app's conversation list) showing
 * past judgments, most recent first. Always visible alongside whichever
 * screen is active -- unlike UploadScreen/CandidateSelectionScreen,
 * this component never unmounts, so it needs an explicit signal to
 * refetch: `refreshKey` is bumped by App.tsx every time a judgment is
 * submitted.
 *
 * Clicking a row opens a centered modal (not an inline expand -- that
 * read as buried in a narrow 260px column) with the full photo +
 * keypoint overlay and a clear min/max/value breakdown. Detail (bbox,
 * keypoints) isn't in the list response, so it's fetched lazily via
 * getGaugeResult() only when a row is actually opened, and cached so
 * re-opening the same row doesn't refetch.
 */
import { useEffect, useState } from "react";
import { getHistory, getGaugeResult, ApiError, type HistoryItem, type GaugeResultResponse } from "../api/client";
import { ImageKeypointOverlay } from "./ImageKeypointOverlay";
import "./HistorySidebar.css";

interface HistorySidebarProps {
  refreshKey: number;
}

export function HistorySidebar({ refreshKey }: HistorySidebarProps) {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [selectedItem, setSelectedItem] = useState<HistoryItem | null>(null);
  const [detailCache, setDetailCache] = useState<Record<string, GaugeResultResponse>>({});
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  useEffect(() => {
    getHistory()
      .then(setItems)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Couldn't load history"));
  }, [refreshKey]);

  async function handleItemClick(item: HistoryItem) {
    setSelectedItem(item);
    setDetailError(null);

    if (!detailCache[item.gauge_id]) {
      setDetailLoading(true);
      try {
        const detail = await getGaugeResult(item.gauge_id);
        setDetailCache((prev) => ({ ...prev, [item.gauge_id]: detail }));
      } catch (err) {
        setDetailError(err instanceof ApiError ? err.message : "Couldn't load this reading");
      } finally {
        setDetailLoading(false);
      }
    }
  }

  const detail = selectedItem ? detailCache[selectedItem.gauge_id] : undefined;

  return (
    <aside className="history-sidebar">
      <div className="history-sidebar__header">
        <span className="history-sidebar__title">History</span>
        {items && <span className="history-sidebar__count">{items.length}</span>}
      </div>

      <div className="history-sidebar__list">
        {error && <p className="history-sidebar__error">{error}</p>}
        {items && items.length === 0 && <p className="history-sidebar__empty">No readings yet</p>}

        {items?.map((item) => (
          <button key={item.gauge_id} className="history-item" onClick={() => handleItemClick(item)}>
            {item.image_url ? (
              <img src={item.image_url} alt="" className="history-item__thumb" />
            ) : (
              <span className="history-item__thumb history-item__thumb--empty" />
            )}
            <div className="history-item__info">
              <div className="history-item__top">
                <span className={`history-item__dot history-item__dot--${item.judgment ?? "none"}`} aria-hidden />
                <span className="history-item__value">{item.computed_value.toFixed(2)}</span>
                <span className="history-item__date">{formatDate(item.created_at)}</span>
              </div>
              <div className="history-item__range">
                Min {item.chosen_min_value} &middot; Max {item.chosen_max_value}
                {item.actual_value !== null && <> &middot; Actual {item.actual_value}</>}
              </div>
            </div>
          </button>
        ))}
      </div>

      {selectedItem && (
        <div className="history-modal-backdrop" onClick={() => setSelectedItem(null)}>
          <div className="history-modal" onClick={(e) => e.stopPropagation()}>
            {selectedItem.image_url ? (
              <ImageKeypointOverlay
                imageUrl={selectedItem.image_url}
                keypoints={detail?.keypoints}
                cropBbox={detail?.bbox}
              />
            ) : (
              <p className="history-sidebar__empty">No photo saved for this reading</p>
            )}

            {detailError && <p className="history-sidebar__error">{detailError}</p>}
            {detailLoading && !detail && <p className="history-sidebar__empty">Loading keypoints…</p>}

            <div className="history-modal__value">
              <span className="history-modal__value-label">Reading</span>
              <span className={`history-modal__value-number history-modal__value-number--${selectedItem.judgment ?? "unclear"}`}>
                {selectedItem.computed_value.toFixed(2)}
              </span>
            </div>

            <div className="history-modal__stats">
              <div className="history-modal__stat">
                <span className="history-modal__stat-label">Min</span>
                <span className="history-modal__stat-value">{selectedItem.chosen_min_value}</span>
              </div>
              <div className="history-modal__stat">
                <span className="history-modal__stat-label">Max</span>
                <span className="history-modal__stat-value">{selectedItem.chosen_max_value}</span>
              </div>
              {selectedItem.actual_value !== null && (
                <div className="history-modal__stat">
                  <span className="history-modal__stat-label">Actual</span>
                  <span className="history-modal__stat-value">{selectedItem.actual_value}</span>
                </div>
              )}
              <div className="history-modal__stat">
                <span className="history-modal__stat-label">Judgment</span>
                <span className={`history-modal__stat-value history-modal__stat-value--${selectedItem.judgment ?? "none"}`}>
                  {selectedItem.judgment ?? "-"}
                </span>
              </div>
            </div>

            <button className="history-modal__back" onClick={() => setSelectedItem(null)}>
              Back to Meter Reader
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}