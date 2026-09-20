/**
 * src/components/PhotoDropzone.tsx
 *
 * v2 of the upload control. v1 was shaped like a circular gauge dial --
 * dropped after feedback that it implied "only round gauges work here",
 * when the backend (gauge_detection.py) actually detects both circular
 * AND rectangular gauges. A viewfinder frame (corner brackets, like a
 * camera's autofocus corners) says "take/drop a photo" without implying
 * anything about the gauge's own shape. Also larger and less severe --
 * the original read as a small, isolated dot on a dark, empty page.
 */
import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import "./PhotoDropzone.css";

interface PhotoDropzoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export function PhotoDropzone({ onFileSelected, disabled = false }: PhotoDropzoneProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    if (!disabled) setIsDragActive(true);
  }

  function handleDragLeave() {
    setIsDragActive(false);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragActive(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) onFileSelected(file);
  }

  function handleClick() {
    if (!disabled) inputRef.current?.click();
  }

  function handleInputChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onFileSelected(file);
  }

  return (
    <div
      className={`photo-dropzone ${isDragActive ? "photo-dropzone--active" : ""} ${disabled ? "photo-dropzone--disabled" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label="Upload a gauge photo"
      onKeyDown={(e) => {
        if (!disabled && (e.key === "Enter" || e.key === " ")) handleClick();
      }}
    >
      {/* Faint dashes in the background -- a texture borrowed from a
          gauge's own scale markings, without being the boundary shape
          itself (that was the problem with the circular version). */}
      <svg viewBox="0 0 100 100" className="photo-dropzone__texture" aria-hidden>
        {Array.from({ length: 24 }).map((_, i) => (
          <line
            key={i}
            x1="50" y1="6" x2="50" y2="12"
            transform={`rotate(${(i / 24) * 360} 50 50)`}
          />
        ))}
      </svg>

      {/* Viewfinder corner brackets */}
      <svg viewBox="0 0 100 100" className="photo-dropzone__brackets" aria-hidden>
        <path d="M18 30 V18 H30" />
        <path d="M70 18 H82 V30" />
        <path d="M82 70 V82 H70" />
        <path d="M30 82 H18 V70" />
      </svg>

      <svg viewBox="0 0 24 24" className="photo-dropzone__icon" aria-hidden>
        <path
          d="M4 8.5A1.5 1.5 0 0 1 5.5 7h2l1-1.5h7L16.5 7h2A1.5 1.5 0 0 1 20 8.5v9A1.5 1.5 0 0 1 18.5 19h-13A1.5 1.5 0 0 1 4 17.5v-9Z"
          fill="none"
          strokeWidth="1.4"
        />
        <circle cx="12" cy="13" r="3.4" fill="none" strokeWidth="1.4" />
      </svg>

      <div className="photo-dropzone__label">
        <span className="photo-dropzone__label-primary">
          {isDragActive ? "Drop it" : "Drop a gauge photo"}
        </span>
        <span className="photo-dropzone__label-secondary">or click to browse</span>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="photo-dropzone__input"
        onChange={handleInputChange}
        disabled={disabled}
      />
    </div>
  );
}