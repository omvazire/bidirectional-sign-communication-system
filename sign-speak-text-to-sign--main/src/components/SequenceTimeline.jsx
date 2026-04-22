/**
 * Sequence Timeline - Shows the breakdown of signs being performed
 */
import React from 'react';
import { Type, Hand, Space } from 'lucide-react';

export default function SequenceTimeline({ sequence, currentIndex }) {
  if (!sequence || sequence.length === 0) {
    return (
      <div className="glass rounded-2xl p-4 lg:p-5" id="sequence-timeline">
        <p className="text-xs text-text-muted text-center py-6">
          Enter text above to see the sign sequence
        </p>
      </div>
    );
  }

  return (
    <div className="glass rounded-2xl p-4 lg:p-5" id="sequence-timeline">
      <h3 className="text-sm font-semibold text-text-secondary mb-3 flex items-center gap-2">
        <Type className="w-4 h-4 text-accent" />
        Sign Sequence
        <span className="text-text-muted font-normal">({sequence.length} signs)</span>
      </h3>
      
      <div className="flex flex-wrap gap-1.5 max-h-40 overflow-y-auto pr-1 custom-scrollbar">
        {sequence.map((step, idx) => {
          const isActive = idx === currentIndex;
          const isDone = idx < currentIndex;
          const isSpace = step.type === 'space';

          if (isSpace) {
            return (
              <div
                key={idx}
                className={`w-6 h-8 flex items-center justify-center rounded 
                  ${isActive ? 'text-accent' : 'text-text-muted/30'}`}
              >
                <Space className="w-3 h-3" />
              </div>
            );
          }

          return (
            <div
              key={idx}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-300
                flex items-center gap-1.5
                ${isActive
                  ? 'bg-primary/25 text-primary-light border border-primary/40 scale-105 shadow-lg shadow-primary/10'
                  : isDone
                    ? 'bg-success/10 text-success/70 border border-success/20'
                    : 'bg-surface-light/40 text-text-muted border border-surface-lighter/20'
                }`}
            >
              {step.type === 'word_sign' && <Hand className="w-3 h-3" />}
              <span>
                {step.type === 'fingerspell' 
                  ? step.letter 
                  : step.type === 'word_sign' 
                    ? (step.isFirst ? step.word : '·') 
                    : ''}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
