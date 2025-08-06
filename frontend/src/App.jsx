import React, { useEffect, useRef, useState } from 'react';

export default function App() {
  const videoRef = useRef(null);
  const socketRef = useRef(null);
  const recorderRef = useRef(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState('Disconnected');
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const wsUrl = `ws://${location.hostname}:8000/ws`;
    socketRef.current = new WebSocket(wsUrl);
    socketRef.current.onopen = () => setStatus('Connected');
    socketRef.current.onclose = () => {
      setStatus('Disconnected');
      stopRecording();
    };
    socketRef.current.onerror = () => setError('WebSocket error');
    socketRef.current.onmessage = ev => {
      const data = JSON.parse(ev.data);
      if (data.transcript) {
        setHistory(h => [...h, data.transcript]);
      }
      if (data.error) setError(data.error);
    };
    return () => socketRef.current?.close();
  }, []);

  const startRecorder = stream => {
    recorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm' });
    recorderRef.current.ondataavailable = ev => {
      if (ev.data.size > 0 && socketRef.current.readyState === 1) {
        socketRef.current.send(ev.data);
      }
    };
    recorderRef.current.start(2000);
    setRecording(true);
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      startRecorder(stream);
    } catch (err) {
      setError('Camera error: ' + err.message);
    }
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    const tracks = videoRef.current?.srcObject?.getTracks();
    tracks && tracks.forEach(t => t.stop());
    videoRef.current.srcObject = null;
    setRecording(false);
  };

  const handleFile = e => {
    const file = e.target.files[0];
    if (!file) return;
    stopRecording();
    const url = URL.createObjectURL(file);
    videoRef.current.srcObject = null;
    videoRef.current.src = url;
    videoRef.current.onloadedmetadata = () => {
      videoRef.current.play();
      const stream = videoRef.current.captureStream();
      startRecorder(stream);
    };
  };

  const clearHistory = () => setHistory([]);

  return (
    <div className="container">
      <h1>Live Transcription</h1>
      <video ref={videoRef} autoPlay playsInline controls />
      <div className="controls">
        <button onClick={recording ? stopRecording : startWebcam}>
          {recording ? 'Stop' : 'Start'} Webcam
        </button>
        <label>
          <span style={{cursor: 'pointer', padding: '0.2rem 0.5rem', border: '1px solid #ccc', borderRadius: '4px'}}>Upload</span>
          <input type="file" accept="video/*" onChange={handleFile} style={{ display: 'none' }} />
        </label>
        <button onClick={clearHistory} disabled={!history.length}>
          Clear
        </button>
      </div>
      <div className="status">Status: {status}</div>
      {error && <div className="status" style={{ color: 'red' }}>{error}</div>}
      <div className="transcript-box">
        <textarea readOnly value={history.join('\n')} />
      </div>
    </div>
  );
}

