/**
 * src/App.tsx
 *
 * Wires the screens together with a minimal state machine, plus a
 * persistent HistorySidebar that never unmounts as screens switch --
 * unlike UploadScreen/CandidateSelectionScreen. Because it's always
 * mounted, it needs an explicit nudge to refetch after something
 * changes the history (a judgment being submitted): historyRefreshKey
 * is bumped in handleFinished, which HistorySidebar watches to know
 * when to reload.
 */
import { useState } from "react";
import { UploadScreen } from "./screens/UploadScreen";
import { CandidateSelectionScreen } from "./screens/CandidateSelectionScreen";
import { HistorySidebar } from "./components/HistorySidebar";
import "./App.css";

type Screen =
  | { name: "upload" }
  | { name: "candidates"; gaugeId: string; imageUrl: string };

function App() {
  const [screen, setScreen] = useState<Screen>({ name: "upload" });
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  function handleFinished() {
    setHistoryRefreshKey((k) => k + 1);
    setScreen({ name: "upload" });
  }

  return (
    <div className="app-shell">
      <HistorySidebar refreshKey={historyRefreshKey} />
      <main className="app-shell__content">
        {screen.name === "candidates" ? (
          <CandidateSelectionScreen
            gaugeId={screen.gaugeId}
            imageUrl={screen.imageUrl}
            onFinished={handleFinished}
          />
        ) : (
          <UploadScreen
            onGaugeReady={(gaugeId, imageUrl) => setScreen({ name: "candidates", gaugeId, imageUrl })}
          />
        )}
      </main>
    </div>
  );
}

export default App;