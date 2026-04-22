import React, { useState, useCallback, useRef } from 'react';
import { Mic, MicOff, Send, RotateCcw, Keyboard, ChevronDown, ChevronUp } from 'lucide-react';

const QUICK_CATEGORIES = [
  {
    label: '👋 Greetings',
    words: ['Hello', 'Hi', 'Bye', 'Goodbye', 'Welcome'],
  },
  {
    label: '💬 Social',
    words: ['Please', 'Thanks', 'Sorry', 'Yes', 'No', 'Maybe', 'OK', 'Excuse'],
  },
  {
    label: '👤 People',
    words: ['Me', 'You', 'Friend', 'Family', 'Mom', 'Dad', 'Brother', 'Sister', 'Baby', 'Teacher'],
  },
  {
    label: '❓ Questions',
    words: ['What', 'Where', 'When', 'Who', 'Why', 'How'],
  },
  {
    label: '🔧 Actions',
    words: ['Want', 'Need', 'Like', 'Love', 'Know', 'Think', 'Help', 'Go', 'Come', 'Wait', 'Stop', 'Eat', 'Drink', 'Sleep', 'Work', 'Learn', 'See', 'Hear'],
  },
  {
    label: '😊 Feelings',
    words: ['Happy', 'Sad', 'Angry', 'Tired', 'Hungry', 'Sick', 'Good', 'Bad'],
  },
  {
    label: '🕐 Time',
    words: ['Now', 'Today', 'Tomorrow', 'Yesterday', 'Morning', 'Night', 'Later', 'Again', 'Always'],
  },
  {
    label: '🏠 Things',
    words: ['Home', 'School', 'Food', 'Water', 'Phone', 'Money', 'Car', 'Book', 'Name'],
  },
  {
    label: '✨ Other',
    words: ['Beautiful', 'Important', 'Different', 'Big', 'Small', 'Fast', 'New', 'More', 'Not', 'Very'],
  },
];

export default function TextInput({ onTranslate, onReset, isPlaying }) {
  const [text, setText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const recognitionRef = useRef(null);

  const handleSubmit = useCallback((e) => {
    e?.preventDefault();
    if (text.trim()) onTranslate(text.trim());
  }, [text, onTranslate]);

  const handleQuickPhrase = useCallback((phrase) => {
    setText(phrase);
    onTranslate(phrase);
  }, [onTranslate]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  }, [handleSubmit]);

  const toggleVoice = useCallback(() => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      alert('Speech recognition not supported.'); return;
    }
    if (isListening) { recognitionRef.current?.stop(); setIsListening(false); return; }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const rec = new SR();
    rec.continuous = false; rec.interimResults = true; rec.lang = 'en-US';
    rec.onresult = (e) => {
      const t = Array.from(e.results).map(r => r[0].transcript).join('');
      setText(t);
    };
    rec.onend = () => { setIsListening(false); };
    rec.onerror = () => setIsListening(false);
    recognitionRef.current = rec; rec.start(); setIsListening(true);
  }, [isListening]);

  const visibleCategories = showAll ? QUICK_CATEGORIES : QUICK_CATEGORIES.slice(0, 4);

  return (
    <div className="glass rounded-2xl p-5 lg:p-6 space-y-4" id="text-input-panel">
      <div className="flex items-center gap-3 mb-1">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
          <Keyboard className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-lg font-display font-semibold text-text-primary">Text Input</h2>
          <p className="text-[11px] text-text-muted">Type, speak, or tap a quick sign</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="relative">
          <textarea id="text-input-field" value={text}
            onChange={(e) => setText(e.target.value)} onKeyDown={handleKeyDown}
            placeholder="Type English text here..." rows={2}
            className="w-full bg-surface rounded-xl px-4 py-3 text-text-primary placeholder:text-text-muted
              border border-surface-lighter/50 focus:border-primary/50 focus:outline-none focus:ring-2
              focus:ring-primary/20 transition-all duration-300 resize-none text-base font-sans" />
          {isListening && (
            <div className="absolute top-3 right-3 flex items-center gap-2">
              <span className="w-2 h-2 bg-error rounded-full animate-pulse" />
              <span className="text-xs text-error font-medium">Listening...</span>
            </div>
          )}
        </div>
        <div className="flex gap-2">
          <button type="submit" disabled={!text.trim() || isPlaying} id="translate-button"
            className="flex-1 bg-gradient-to-r from-primary to-primary-dark hover:from-primary-light
              hover:to-primary disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold
              py-2.5 px-4 rounded-xl transition-all duration-300 flex items-center justify-center gap-2
              shadow-lg shadow-primary/20 hover:shadow-primary/40 active:scale-[0.98]">
            <Send className="w-4 h-4" />Translate
          </button>
          <button type="button" onClick={toggleVoice} id="voice-input-button"
            className={`p-2.5 rounded-xl transition-all duration-300 ${isListening
              ? 'bg-error/20 text-error border border-error/30'
              : 'bg-surface-light text-text-secondary hover:text-accent border border-surface-lighter/50 hover:border-accent/30'}`}>
            {isListening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </button>
          <button type="button" onClick={() => { setText(''); onReset(); }} id="reset-button"
            className="p-2.5 rounded-xl bg-surface-light text-text-secondary hover:text-warning
              border border-surface-lighter/50 hover:border-warning/30 transition-all duration-300">
            <RotateCcw className="w-5 h-5" />
          </button>
        </div>
      </form>

      <div className="space-y-2.5">
        <p className="text-xs text-text-muted font-medium uppercase tracking-wider">Quick Signs</p>
        {visibleCategories.map((cat) => (
          <div key={cat.label}>
            <p className="text-[11px] text-text-muted/70 mb-1">{cat.label}</p>
            <div className="flex flex-wrap gap-1">
              {cat.words.map((word) => (
                <button key={word} onClick={() => handleQuickPhrase(word)}
                  className="px-2.5 py-1 text-[11px] font-medium rounded-lg bg-surface-light/50
                    text-text-secondary hover:bg-primary/15 hover:text-primary-light border
                    border-surface-lighter/20 hover:border-primary/30 transition-all duration-200 active:scale-95">
                  {word}
                </button>
              ))}
            </div>
          </div>
        ))}
        <button onClick={() => setShowAll(!showAll)}
          className="w-full flex items-center justify-center gap-1 py-1.5 text-xs text-text-muted
            hover:text-primary-light transition-colors">
          {showAll ? <><ChevronUp className="w-3.5 h-3.5" />Show Less</> : <><ChevronDown className="w-3.5 h-3.5" />Show All ({QUICK_CATEGORIES.length} categories)</>}
        </button>
      </div>
    </div>
  );
}
