import { useState, useRef } from 'react'
import axios from 'axios'

const API = 'http://localhost:8000'

const EMOTION_EMOJIS = { calm: '🌊', energetic: '⚡', happy: '☀️', sad: '🌧️' }

function EmotionScores({ scores }) {
  return (
    <div className="emotion-scores">
      {Object.entries(scores).map(([emotion, score]) => (
        <div key={emotion} className="emotion-score-row">
          <span className="emotion-score-label">{emotion}</span>
          <div className="emotion-score-track">
            <div className={`emotion-score-fill fill-${emotion}`} style={{ width: `${score * 100}%` }} />
          </div>
          <span className="emotion-score-value">{score.toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

/**
 * Circumplex visualization of the continuous valence-arousal mood coordinates.
 * Valence maps to the horizontal axis (left = gloomy, right = uplifting),
 * arousal maps to the vertical axis (bottom = calm, top = energetic).
 */
function MoodPlane({ mood }) {
  if (!mood) return null

  const size = 240
  const pad  = 24
  const half = size / 2
  const r    = half - pad

  // Map (-1..1) coordinates into SVG pixel space.
  // SVG y grows downward, so invert arousal.
  const px = half + mood.valence * r
  const py = half - mood.arousal * r

  return (
    <div className="mood-plane">
      <div className="mood-plane-header">
        <div className="mood-plane-title">Mood Circumplex</div>
        <div className="mood-plane-subtitle">
          Continuous valence x arousal - v1.1 dimensional layer
        </div>
      </div>

      <div className="mood-plane-body">
        <svg width={size} height={size} className="mood-plane-svg">
          <rect x="1" y="1" width={size - 2} height={size - 2} rx="14"
            fill="var(--bg-primary)" stroke="var(--border-light)" />
          <circle cx={half} cy={half} r={r}
            fill="none" stroke="var(--border-light)" strokeDasharray="2 4" />
          <line x1={pad} y1={half} x2={size - pad} y2={half}
            stroke="var(--border-light)" />
          <line x1={half} y1={pad} x2={half} y2={size - pad}
            stroke="var(--border-light)" />

          <text x={size - pad + 2} y={half + 4} fontSize="9"
            fill="var(--text-muted)" textAnchor="start">+ valence</text>
          <text x={pad - 2} y={half + 4} fontSize="9"
            fill="var(--text-muted)" textAnchor="end">- valence</text>
          <text x={half} y={pad - 4} fontSize="9"
            fill="var(--text-muted)" textAnchor="middle">+ arousal</text>
          <text x={half} y={size - pad + 12} fontSize="9"
            fill="var(--text-muted)" textAnchor="middle">- arousal</text>

          <text x={size - pad - 4} y={pad + 8} fontSize="10"
            fill="var(--energetic)" textAnchor="end" fontWeight="600">Excited</text>
          <text x={pad + 4} y={pad + 8} fontSize="10"
            fill="#ff5577" textAnchor="start" fontWeight="600">Tense</text>
          <text x={pad + 4} y={size - pad - 4} fontSize="10"
            fill="var(--sad)" textAnchor="start" fontWeight="600">Melancholic</text>
          <text x={size - pad - 4} y={size - pad - 4} fontSize="10"
            fill="var(--calm)" textAnchor="end" fontWeight="600">Serene</text>

          <circle cx={px} cy={py} r="9"
            fill="var(--accent)" stroke="var(--bg-primary)" strokeWidth="2"
            style={{ filter: 'drop-shadow(0 0 6px var(--accent-glow))' }} />
        </svg>

        <div className="mood-plane-details">
          <div className="mood-chip mood-chip-primary">
            {mood.nuanced_tag}
          </div>
          <div className="mood-plane-row">
            <span className="mood-plane-label">Quadrant</span>
            <span className="mood-plane-value">{mood.quadrant}</span>
          </div>
          <div className="mood-plane-row">
            <span className="mood-plane-label">Valence</span>
            <span className="mood-plane-value">
              {mood.valence >= 0 ? '+' : ''}{mood.valence.toFixed(2)}
            </span>
          </div>
          <div className="mood-plane-row">
            <span className="mood-plane-label">Arousal</span>
            <span className="mood-plane-value">
              {mood.arousal >= 0 ? '+' : ''}{mood.arousal.toFixed(2)}
            </span>
          </div>
          <div className="mood-plane-row">
            <span className="mood-plane-label">Intensity</span>
            <span className="mood-plane-value">
              {(mood.confidence * 100).toFixed(0)}%
            </span>
          </div>
          {mood.rationale && (
            <div className="mood-plane-rationale">{mood.rationale}</div>
          )}
        </div>
      </div>
    </div>
  )
}


function TrackResult({ result, onFeedback, fileNum }) {
  const [rated, setRated] = useState(null)

  const handleFeedback = async (label) => {
    await axios.post(`${API}/feedback`, {
      path: result.path,
      label,
      emotion_label: result.emotion,
      emotion_scores: result.scores,
      vector: result.vector,
    })
    setRated(label)
    onFeedback()
  }

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <span style={{ fontSize: 28 }}>{EMOTION_EMOJIS[result.emotion] || '🎵'}</span>
            <div>
              <div style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: 20 }}>
                File {fileNum}
              </div>
              <span className={`track-emotion-badge badge-${result.emotion}`}>
                {result.emotion}
              </span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className={`btn-icon btn-like ${rated === 'like' ? 'active' : ''}`}
            onClick={() => handleFeedback('like')}
            disabled={!!rated}
            title="Like"
          >
            {rated === 'like' ? '♥' : '♡'}
          </button>
          <button
            className={`btn-icon btn-dislike ${rated === 'dislike' ? 'active' : ''}`}
            onClick={() => handleFeedback('dislike')}
            disabled={!!rated}
            title="Dislike"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="stat-row">
        <div className="stat-pill">
          <span className="stat-pill-label">Sample Rate</span>
          <span className="stat-pill-value">{result.sr?.toLocaleString()}</span>
        </div>
        <div className="stat-pill">
          <span className="stat-pill-label">Duration</span>
          <span className="stat-pill-value">{result.duration_sec?.toFixed(1)}s</span>
        </div>
        <div className="stat-pill">
          <span className="stat-pill-label">BPM</span>
          <span className="stat-pill-value">{result.tempo?.toFixed(0) || '—'}</span>
        </div>
        <div className="stat-pill">
          <span className="stat-pill-label">Vector Dim</span>
          <span className="stat-pill-value">{result.vector_dim}</span>
        </div>
      </div>

      <hr className="divider" style={{ margin: '16px 0' }} />
      <EmotionScores scores={result.scores} />

      {result.mood && (
        <>
          <hr className="divider" style={{ margin: '16px 0' }} />
          <MoodPlane mood={result.mood} />
        </>
      )}

      {rated && (
        <div style={{ marginTop: 12, fontSize: 12, color: 'var(--accent)' }}>
          {rated === 'like' ? '♥ Added to your profile' : '✕ Noted as dislike'}
        </div>
      )}
    </div>
  )
}

export default function Analyze({ onFeedback }) {
  const [file1, setFile1] = useState(null)
  const [file2, setFile2] = useState(null)
  const [result1, setResult1] = useState(null)
  const [result2, setResult2] = useState(null)
  const [similarity, setSimilarity] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const input1Ref = useRef()
  const input2Ref = useRef()

  const analyzeFile = async (file) => {
    const form = new FormData()
    form.append('file', file)
    const res = await axios.post(`${API}/analyze`, form)
    return res.data
  }

  const handleAnalyze = async () => {
    if (!file1) return
    setLoading(true)
    setSimilarity(null)
    try {
      const r1 = await analyzeFile(file1)
      setResult1(r1)

      if (file2) {
        const r2 = await analyzeFile(file2)
        setResult2(r2)

        const form = new FormData()
        form.append('file1', file1)
        form.append('file2', file2)
        const simRes = await axios.post(`${API}/similarity`, form)
        setSimilarity(simRes.data.similarity)
      }
    } finally {
      setLoading(false)
    }
  }

  const similarityVerdict = (s) => {
    if (s >= 0.90) return 'Very similar — close in MFCC feature space'
    if (s >= 0.75) return 'Somewhat similar — shared acoustic traits'
    return 'Less similar — different timbre and energy patterns'
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Analyze Audio</h1>
        <p className="page-subtitle">Upload audio files to extract MFCC features, map emotions, and compare similarity.</p>
      </div>

      {/* Upload zones */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        <div>
          <div
            className={`upload-zone ${dragging ? 'dragging' : ''}`}
            onClick={() => input1Ref.current.click()}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); setFile1(e.dataTransfer.files[0]); setResult1(null) }}
          >
            <div className="upload-icon">⟁</div>
            <div className="upload-text">Audio File 1</div>
            <div className="upload-hint">WAV or MP3 · Required</div>
            <input ref={input1Ref} type="file" accept=".wav,.mp3" style={{ display: 'none' }}
              onChange={e => { setFile1(e.target.files[0]); setResult1(null) }} />
          </div>
          {file1 && (
            <div className="file-tag">
              🎵 {file1.name}
              <span className="file-tag-remove" onClick={() => { setFile1(null); setResult1(null) }}>✕</span>
            </div>
          )}
        </div>

        <div>
          <div
            className="upload-zone"
            onClick={() => input2Ref.current.click()}
            onDragOver={e => { e.preventDefault() }}
            onDrop={e => { e.preventDefault(); setFile2(e.dataTransfer.files[0]); setResult2(null) }}
          >
            <div className="upload-icon">⟁</div>
            <div className="upload-text">Audio File 2</div>
            <div className="upload-hint">WAV or MP3 · Optional — for similarity</div>
            <input ref={input2Ref} type="file" accept=".wav,.mp3" style={{ display: 'none' }}
              onChange={e => { setFile2(e.target.files[0]); setResult2(null) }} />
          </div>
          {file2 && (
            <div className="file-tag">
              🎵 {file2.name}
              <span className="file-tag-remove" onClick={() => { setFile2(null); setResult2(null); setSimilarity(null) }}>✕</span>
            </div>
          )}
        </div>
      </div>

      <button className="btn btn-primary" onClick={handleAnalyze} disabled={!file1 || loading}
        style={{ marginBottom: 32 }}>
        {loading ? '⟳ Analyzing...' : '⟁ Analyze'}
      </button>

      {loading && <div className="loading"><div className="spinner" /> Extracting features...</div>}

      {result1 && <TrackResult result={result1} onFeedback={onFeedback} fileNum={1} />}
      {result2 && <TrackResult result={result2} onFeedback={onFeedback} fileNum={2} />}

      {similarity !== null && (
        <div className="similarity-display">
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
              Cosine Similarity
            </div>
            <div className="similarity-score">{similarity.toFixed(4)}</div>
          </div>
          <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: 20 }}>
            <div className="similarity-verdict">{similarityVerdict(similarity)}</div>
          </div>
        </div>
      )}
    </div>
  )
}
