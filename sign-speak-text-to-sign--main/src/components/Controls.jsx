/**
 * Playback Controls - Play, Pause, Speed, Progress
 */
import React from 'react';
import { Play, Pause, SkipBack, Gauge, Zap } from 'lucide-react';

const SPEED_OPTIONS = [0.5, 0.75, 1, 1.25, 1.5, 2];

export default function Controls({
  isPlaying,
  speed,
  progress,
  currentLabel,
  currentIndex,
  totalSteps,
  onPlay,
  onPause,
  onReset,
  onSpeedChange,
}) {
  return (
    <div className="glass rounded-2xl p-4 lg:p-5 space-y-4" id="playback-controls">
      {/* Current Sign Display */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-surface/60 border border-surface-lighter/30">
          <Zap className="w-4 h-4 text-accent" />
          <span className="text-sm font-medium text-text-primary truncate max-w-[200px]">
            {currentLabel || 'Ready to translate'}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="space-y-1.5">
        <div className="h-1.5 bg-surface rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all duration-150"
            style={{ width: `${Math.max(progress * 100, 0)}%` }}
          />
        </div>
        <div className="flex justify-between text-[11px] text-text-muted">
          <span>{currentIndex + 1} / {totalSteps || 0}</span>
          <span>{Math.round(progress * 100)}%</span>
        </div>
      </div>

      {/* Playback Buttons */}
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={onReset}
          id="controls-reset"
          className="p-2 rounded-lg bg-surface-light/60 text-text-secondary
                     hover:text-text-primary hover:bg-surface-lighter/60
                     transition-all duration-200 active:scale-90"
          title="Reset"
        >
          <SkipBack className="w-4 h-4" />
        </button>

        <button
          onClick={isPlaying ? onPause : onPlay}
          disabled={totalSteps === 0}
          id="controls-play-pause"
          className="p-3.5 rounded-xl bg-gradient-to-r from-primary to-primary-dark
                     hover:from-primary-light hover:to-primary
                     disabled:opacity-40 disabled:cursor-not-allowed
                     text-white shadow-lg shadow-primary/25
                     hover:shadow-primary/40 transition-all duration-300
                     active:scale-95"
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
        </button>
      </div>

      {/* Speed Control */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-text-muted flex items-center gap-1.5">
            <Gauge className="w-3.5 h-3.5" />
            Speed
          </span>
          <span className="text-xs font-semibold text-primary-light">{speed}x</span>
        </div>
        <div className="flex gap-1">
          {SPEED_OPTIONS.map((s) => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              className={`flex-1 py-1.5 text-[11px] font-medium rounded-lg transition-all duration-200
                ${speed === s
                  ? 'bg-primary/20 text-primary-light border border-primary/30'
                  : 'bg-surface-light/40 text-text-muted hover:text-text-secondary border border-transparent hover:border-surface-lighter/30'
                }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
