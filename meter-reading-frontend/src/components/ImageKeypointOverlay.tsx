/**
 * src/components/ImageKeypointOverlay.tsx
 *
 * Shows the uploaded photo -- optionally cropped to Stage 1's detected
 * bbox, optionally with the 4 keypoints drawn on top. Used on the
 * upload screen (no bbox/keypoints yet), the candidate-selection
 * screen, and the result screen.
 *
 * Rendering is done via an SVG <image>, not a plain <img>, specifically
 * so cropping is possible: the outer <svg>'s viewBox can be set to just
 * the bbox region (in the image's own pixel coordinates) instead of the
 * full photo, which crops-and-zooms automatically -- no canvas, no
 * separately-stored cropped image file on the backend. keypoints_global
 * values are already in this same original-image pixel space (see
 * key_points.map_points_to_original on the backend), so they plot
 * correctly whether the current view is cropped or not, with zero
 * coordinate math on this end.
 */
import { useState, type SyntheticEvent } from "react";
import type { GaugeKeypoints } from "../api/client";
import "./ImageKeypointOverlay.css";

interface ImageKeypointOverlayProps {
  imageUrl: string;
  keypoints?: GaugeKeypoints | null;
  cropBbox?: [number, number, number, number] | null;
}

export function ImageKeypointOverlay({ imageUrl, keypoints, cropBbox }: ImageKeypointOverlayProps) {
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);

  function handleLoad(e: SyntheticEvent<HTMLImageElement>) {
    const img = e.currentTarget;
    setNatural({ w: img.naturalWidth, h: img.naturalHeight });
  }

  if (!natural) {
    // A visually-hidden <img> purely to learn the photo's natural pixel
    // size before anything can be drawn -- browsers still fire onLoad
    // for a 0x0-rendered image, so this never becomes visible itself.
    return (
      <div className="image-overlay">
        <img src={imageUrl} alt="" className="image-overlay__sizer" onLoad={handleLoad} />
      </div>
    );
  }

  const [vx, vy, vw, vh] = cropBbox ?? [0, 0, natural.w, natural.h];
  const regionWidth = vw; // used to scale dot/stroke/font sizes to whatever's currently in view

  const dotRadius = regionWidth * 0.014;
  const strokeWidth = regionWidth * 0.005;
  const fontSize = regionWidth * 0.032;

  return (
    <div
      className="image-overlay"
      style={{ aspectRatio: `${vw} / ${vh}` }}
    >
      <svg
        className="image-overlay__svg"
        viewBox={`${vx} ${vy} ${vw} ${vh}`}
        preserveAspectRatio="xMidYMid slice"
      >
        <image href={imageUrl} x={0} y={0} width={natural.w} height={natural.h} />

        {keypoints && (
          <>
            <line
              x1={keypoints.center[0]} y1={keypoints.center[1]}
              x2={keypoints.tip[0]} y2={keypoints.tip[1]}
              className="image-overlay__needle"
              strokeWidth={strokeWidth}
            />
            <Point x={keypoints.center[0]} y={keypoints.center[1]} r={dotRadius} className="image-overlay__dot--center" />
            <Point x={keypoints.min[0]} y={keypoints.min[1]} r={dotRadius} className="image-overlay__dot--min" label="min" fontSize={fontSize} />
            <Point x={keypoints.max[0]} y={keypoints.max[1]} r={dotRadius} className="image-overlay__dot--max" label="max" fontSize={fontSize} />
            <Point x={keypoints.tip[0]} y={keypoints.tip[1]} r={dotRadius * 0.8} className="image-overlay__dot--tip" />
          </>
        )}
      </svg>
    </div>
  );
}

interface PointProps {
  x: number;
  y: number;
  r: number;
  className: string;
  label?: string;
  fontSize?: number;
}

function Point({ x, y, r, className, label, fontSize }: PointProps) {
  return (
    <g>
      <circle cx={x} cy={y} r={r} className={`image-overlay__dot ${className}`} />
      {label && (
        <text x={x + r * 1.8} y={y + r * 0.6} fontSize={fontSize} className={`image-overlay__label ${className}`}>
          {label}
        </text>
      )}
    </g>
  );
}