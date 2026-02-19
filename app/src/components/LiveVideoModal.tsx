import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Video, VideoOff, Scan } from "lucide-react";
import { Button } from "./ui/button";
import { toast } from "sonner";

interface LiveVideoModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (imageBase64: string) => void;
}

export function LiveVideoModal({ isOpen, onClose, onCapture }: LiveVideoModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDevice, setSelectedDevice] = useState<string>("");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);

  // Get available cameras
  useEffect(() => {
    if (!isOpen) return;

    const getDevices = async () => {
      try {
        // Request permission first to get labels
        await navigator.mediaDevices.getUserMedia({ video: true });
        const deviceList = await navigator.mediaDevices.enumerateDevices();
        const cameras = deviceList.filter(d => d.kind === 'videoinput');
        setDevices(cameras);
        if (cameras.length > 0 && !selectedDevice) {
          setSelectedDevice(cameras[0].deviceId);
        }
      } catch (err) {
        console.error("Failed to enumerate devices:", err);
      }
    };

    getDevices();
  }, [isOpen, selectedDevice]);

  // Start/stop stream
  useEffect(() => {
    if (!isOpen) {
      stopStream();
      return;
    }

    if (selectedDevice) {
      startStream();
    }

    return () => {
      stopStream();
    };
  }, [isOpen, selectedDevice]);

  const startStream = async () => {
    try {
      setError(null);
      
      const constraints: MediaStreamConstraints = {
        video: selectedDevice 
          ? { deviceId: { exact: selectedDevice } }
          : { facingMode: "user" },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsStreaming(true);
        };
      }
    } catch (err) {
      console.error("Failed to start video stream:", err);
      setError("Could not access camera. Please check permissions.");
      setIsStreaming(false);
    }
  };

  const stopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  };

  const handleCapture = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame
    ctx.drawImage(video, 0, 0);

    // Convert to base64
    const imageBase64 = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
    
    onCapture(imageBase64);
    toast.success("Image captured from video feed");
  }, [isStreaming, onCapture]);

  const toggleStream = () => {
    if (isStreaming) {
      stopStream();
    } else {
      startStream();
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={(e) => {
            if (e.target === e.currentTarget) onClose();
          }}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="bg-background border rounded-xl overflow-hidden max-w-4xl w-full shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-muted/50">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Video className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold">Live Video Feed</h3>
                  <p className="text-xs text-muted-foreground">
                    {isStreaming ? "Streaming active" : "Stream paused"}
                  </p>
                </div>
              </div>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="w-5 h-5" />
              </Button>
            </div>

            {/* Camera Selection */}
            {devices.length > 1 && (
              <div className="p-3 border-b bg-muted/30">
                <select
                  value={selectedDevice}
                  onChange={(e) => setSelectedDevice(e.target.value)}
                  className="w-full p-2 text-sm bg-background border rounded-lg"
                >
                  {devices.map((device, i) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera ${i + 1}`}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Video Area */}
            <div className="relative aspect-video bg-black flex items-center justify-center">
              {error ? (
                <div className="text-center p-8">
                  <VideoOff className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">{error}</p>
                  <Button 
                    variant="outline" 
                    className="mt-4"
                    onClick={startStream}
                  >
                    Retry
                  </Button>
                </div>
              ) : (
                <>
                  <video
                    ref={videoRef}
                    className="w-full h-full object-contain"
                    playsInline
                    muted
                  />
                  
                  {/* Overlay when not streaming */}
                  {!isStreaming && !error && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                      <div className="text-center">
                        <VideoOff className="w-12 h-12 text-muted-foreground mx-auto mb-2" />
                        <p className="text-muted-foreground">Stream paused</p>
                      </div>
                    </div>
                  )}

                  {/* Recording indicator */}
                  {isStreaming && (
                    <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/60 px-3 py-1.5 rounded-full">
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                      <span className="text-xs text-white font-medium">LIVE</span>
                    </div>
                  )}
                </>
              )}

              {/* Hidden canvas for capture */}
              <canvas ref={canvasRef} className="hidden" />
            </div>

            {/* Controls */}
            <div className="p-4 border-t bg-muted/30">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <Button
                    variant={isStreaming ? "destructive" : "default"}
                    onClick={toggleStream}
                    className="gap-2"
                  >
                    {isStreaming ? (
                      <>
                        <VideoOff className="w-4 h-4" />
                        Stop Stream
                      </>
                    ) : (
                      <>
                        <Video className="w-4 h-4" />
                        Start Stream
                      </>
                    )}
                  </Button>
                </div>

                <Button
                  onClick={handleCapture}
                  disabled={!isStreaming}
                  className="gap-2"
                  size="lg"
                >
                  <Scan className="w-4 h-4" />
                  Capture & Analyze
                </Button>
              </div>

              <p className="mt-3 text-xs text-muted-foreground text-center">
                The AI can see your video feed in real-time. Click "Capture & Analyze" to send the current frame for analysis.
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
