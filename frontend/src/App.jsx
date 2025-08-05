import React, { useEffect, useRef, useState } from 'react';

export default function App() {
  const videoRef = useRef(null);
  const socketRef = useRef(null);
  const recorderRef = useRef(null);
  const [transcript, setTranscript] = useState('');
  const [status, setStatus] = useState('Connecting...');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const wsUrl = `ws://${location.hostname}:8000/ws`;
    socketRef.current = new WebSocket(wsUrl);
    socketRef.current.onopen = () => setStatus('Connected');
    socketRef.current.onclose = () => {
      setStatus('Disconnected');
      setLoading(false);
    };
    socketRef.current.onerror = () => setError('WebSocket error');
    socketRef.current.onmessage = ev => {
      const data = JSON.parse(ev.data);
      setTranscript(data.transcript || '');
      setError(data.error || '');
      setLoading(false);
    };
    return () => socketRef.current?.close();
  }, []);

  const setupRecorder = stream => {
    recorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm' });
    recorderRef.current.ondataavailable = ev => {
      if (ev.data.size > 0 && socketRef.current.readyState === 1) {
        setLoading(true);
        socketRef.current.send(ev.data);
      }
    };
    recorderRef.current.start(2000);
  };

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        videoRef.current.srcObject = stream;
        setupRecorder(stream);
      })
      .catch(err => setError('Camera error: ' + err.message));
  }, []);

  const handleFile = e => {
    const file = e.target.files[0];
    if (!file) return;
    if (recorderRef.current) recorderRef.current.stop();
    videoRef.current.srcObject = null;
    videoRef.current.src = URL.createObjectURL(file);
    videoRef.current.onloadedmetadata = () => {
      videoRef.current.play();
      const stream = videoRef.current.captureStream();
      setupRecorder(stream);
    };
  };

  return (
    <div style={{ textAlign: 'center', fontFamily: 'sans-serif' }}>
      <h1>Live Transcription</h1>
      <video ref={videoRef} autoPlay controls playsInline width="320" height="240" style={{ border: '1px solid #ccc' }} />
      <div>
        <input type="file" accept="video/*" onChange={handleFile} />
      </div>
      <p>Status: {status}</p>
      {loading && <p>Transcribing...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <p>{transcript}</p>
    </div>
  );
}
