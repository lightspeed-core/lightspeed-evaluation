import { useState } from 'react'

export default function CollapsiblePanel({ title, children, defaultOpen = true, tooltip }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="chart-card collapsible">
      <div className="collapsible-header" onClick={() => setOpen(o => !o)}>
        <div className="collapsible-title-row">
          <h3>{title}</h3>
          {tooltip && (
            <span className="info-icon-wrapper" onClick={e => e.stopPropagation()}>
              <svg
                className="info-icon"
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="16" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12.01" y2="8" />
              </svg>
              <span className="info-tooltip">{tooltip}</span>
            </span>
          )}
        </div>
        <svg
          className={`collapse-arrow ${open ? 'expanded' : ''}`}
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="18 15 12 9 6 15" />
        </svg>
      </div>
      {open && children}
    </div>
  )
}
