'use client';

import { useEffect, useState } from 'react';

interface Props {
  open: boolean;
  title: string;
  description?: string;
  initialValue?: string;
  placeholder?: string;
  submitLabel?: string;
  extra?: React.ReactNode;
  onClose: () => void;
  onSubmit: (value: string) => void;
}

const NameInputModal: React.FC<Props> = ({
  open,
  title,
  description,
  initialValue = '',
  placeholder = 'Untitled',
  submitLabel = 'Save',
  extra,
  onClose,
  onSubmit,
}) => {
  const [value, setValue] = useState(initialValue);

  useEffect(() => {
    if (open) setValue(initialValue);
  }, [open, initialValue]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
  };

  return (
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="name-input-modal-title"
    >
      <button
        type="button"
        aria-label="Close"
        onClick={onClose}
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
      />
      <div className="relative w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl">
        <div className="flex items-center justify-between">
          <h2
            id="name-input-modal-title"
            className="text-lg font-semibold tracking-tight text-foreground"
          >
            {title}
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-foreground/50 transition hover:bg-surface-muted hover:text-foreground"
            aria-label="Close"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4"
              aria-hidden="true"
            >
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
          </button>
        </div>

        {description && <p className="mt-2 text-sm text-foreground/65">{description}</p>}

        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          <div>
            <label
              htmlFor="name-input-modal-value"
              className="block text-xs font-semibold uppercase tracking-[0.12em] text-foreground/60"
            >
              Name
            </label>
            <input
              id="name-input-modal-value"
              type="text"
              autoFocus
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onFocus={(e) => e.target.select()}
              placeholder={placeholder}
              className="mt-1.5 block w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-foreground shadow-sm transition placeholder:text-foreground/40 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30"
            />
          </div>

          {extra}

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="inline-flex items-center justify-center rounded-lg border border-border bg-surface px-4 py-2 text-sm font-semibold text-foreground/80 transition hover:border-border-strong hover:bg-surface-muted"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!value.trim()}
              className="inline-flex items-center justify-center rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-accent-foreground transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
            >
              {submitLabel}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NameInputModal;
