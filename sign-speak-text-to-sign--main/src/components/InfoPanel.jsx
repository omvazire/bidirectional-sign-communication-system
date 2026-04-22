import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp, CheckCircle } from 'lucide-react';
import { WORD_SIGNS } from '../data/aslPoses';

export default function InfoPanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const supportedWords = Object.keys(WORD_SIGNS);

  return (
    <div className="glass rounded-2xl p-4 lg:p-5" id="info-panel">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between"
      >
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-accent" />
          <span className="text-sm font-semibold text-text-secondary">
            How it works
          </span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-text-muted" />
        ) : (
          <ChevronDown className="w-4 h-4 text-text-muted" />
        )}
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-3 text-xs text-text-muted">
          <div className="flex items-start gap-2 p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <CheckCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span className="font-medium leading-relaxed">Poses verified against Kaggle WLASL (Word-Level American Sign Language) dataset for accuracy.</span>
          </div>

          <div className="space-y-2">
            <p className="text-text-secondary font-medium">✨ Features:</p>
            <ul className="space-y-1 ml-4">
              <li>• Type any English text to see ASL translation</li>
              <li>• Known words use dataset-verified authentic gestures</li>
              <li>• Unknown words are fingerspelled letter by letter</li>
              <li>• Voice input supported (click mic button)</li>
              <li>• Drag to rotate the 3D hand view</li>
              <li>• Scroll to zoom in/out</li>
            </ul>
          </div>

          <div className="space-y-2">
            <p className="text-text-secondary font-medium">
              🤟 Word Signs ({supportedWords.length} supported):
            </p>
            <div className="flex flex-wrap gap-1">
              {supportedWords.map((word) => (
                <span
                  key={word}
                  className="px-2 py-0.5 rounded text-[10px] bg-accent/10 text-accent/80 border border-accent/15"
                >
                  {word}
                </span>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-text-secondary font-medium">🔤 Fingerspelling:</p>
            <p>All 26 letters (A-Z) and numbers (0-9) are supported</p>
          </div>
        </div>
      )}
    </div>
  );
}
