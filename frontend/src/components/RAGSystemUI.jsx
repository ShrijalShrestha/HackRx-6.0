import React, { useEffect, useState, useRef } from "react";
import { Input } from "./ui/Input";
import { Textarea } from "./ui/Textarea";
import { Button } from "./ui/Button";
import { Card } from "./ui/Card";

const API_BASE = "http://localhost:8000";

export default function RAGSystemUI() {
  const [systemStatus, setSystemStatus] = useState("Checking system status...");
  const [stats, setStats] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [urlInput, setUrlInput] = useState("");
  const [timeout, setTimeoutVal] = useState(30);
  const [urlUploadStatus, setUrlUploadStatus] = useState("");
  const [query, setQuery] = useState("");
  const [queryResult, setQueryResult] = useState(null);
  const fileInputRef = useRef();

  useEffect(() => {
    checkSystemHealth();
    fetchStats();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      const data = await response.json();
      setSystemStatus(
        data.status === "healthy"
          ? "✅ System is healthy and ready"
          : "⚠️ System is degraded"
      );
    } catch (err) {
      setSystemStatus("❌ Cannot connect to API server");
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileSelect = (e) => {
    setSelectedFiles(Array.from(e.target.files));
  };

  const uploadDocuments = async () => {
    if (selectedFiles.length === 0) return;
    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("files", file));
    setUploadStatus("Uploading...");

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setUploadStatus(`✅ ${data.files_processed} files processed in ${data.processing_time.toFixed(2)}s`);
      fetchStats();
    } catch (err) {
      setUploadStatus("❌ Upload failed");
    }
  };

  const uploadFromUrls = async () => {
    const urls = urlInput
      .split("\n")
      .map((url) => url.trim())
      .filter((url) => url);

    setUrlUploadStatus("Uploading from URLs...");
    try {
      const res = await fetch(`${API_BASE}/upload-urls`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls, timeout }),
      });
      const data = await res.json();
      setUrlUploadStatus(`✅ ${data.files_processed} files processed in ${data.processing_time.toFixed(2)}s`);
      fetchStats();
    } catch (err) {
      setUrlUploadStatus("❌ URL upload failed");
    }
  };

  const handleQuery = async () => {
    setQueryResult("Searching...");
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, max_results: 3, score_threshold: 0.1 }),
      });
      const data = await res.json();
      setQueryResult(data);
    } catch (err) {
      setQueryResult("❌ Query failed");
    }
  };

  return (
    <div className="p-6 max-w-5xl mx-auto font-sans space-y-6">
      <div className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-center rounded-2xl py-6">
        <h1 className="text-4xl font-bold">🤖 RAG System API</h1>
        <p className="text-lg mt-2">Document Processing & Semantic Search Interface</p>
      </div>

      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">📊 System Status</h2>
        <div className="bg-blue-100 text-blue-700 p-3 rounded mb-4">{systemStatus}</div>
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Stat label="Documents" value={stats.total_documents || 0} />
            <Stat label="Chunks" value={stats.total_chunks || 0} />
            <Stat label="Vectors" value={stats.vector_count || 0} />
            <Stat label="Status" value={stats.system_ready ? "Ready" : "Not Ready"} />
          </div>
        )}
      </Card>

      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">📁 Upload Documents</h2>
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          multiple
          accept=".pdf,.docx,.txt,.md,.eml"
          onChange={handleFileSelect}
        />
        <Button onClick={() => fileInputRef.current.click()}>Choose Files</Button>
        {selectedFiles.length > 0 && (
          <>
            <ul className="mt-4 space-y-1 text-sm text-gray-700">
              {selectedFiles.map((f, i) => (
                <li key={i}>📄 {f.name}</li>
              ))}
            </ul>
            <Button onClick={uploadDocuments} className="mt-3">
              Upload Documents
            </Button>
          </>
        )}
        {uploadStatus && <p className="mt-2 text-sm text-gray-600">{uploadStatus}</p>}
      </Card>

      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">🌐 Process Documents from URLs</h2>
        <Textarea
          value={urlInput}
          onChange={(e) => setUrlInput(e.target.value)}
          placeholder="Enter one URL per line"
        />
        <div className="flex items-center gap-2 mt-3">
          <Input
            type="number"
            value={timeout}
            onChange={(e) => setTimeoutVal(e.target.value)}
            className="w-20"
          />
          <span className="text-sm text-gray-500">seconds</span>
          <Button onClick={uploadFromUrls}>Download & Process URLs</Button>
        </div>
        {urlUploadStatus && <p className="mt-2 text-sm text-gray-600">{urlUploadStatus}</p>}
      </Card>

      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">🔍 Search & Query</h2>
        <div className="flex gap-2">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your documents..."
            onKeyDown={(e) => e.key === "Enter" && handleQuery()}
          />
          <Button onClick={handleQuery}>Search</Button>
        </div>
        {queryResult && typeof queryResult === "string" && (
          <p className="mt-3 text-green-500">{queryResult}</p>
        )}

        {queryResult && typeof queryResult === "object" && queryResult.answer && (
          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded mt-4 mb-6">
            <h3 className="font-semibold text-blue-700 mb-2">📘 Answer:</h3>
            <p className="text-gray-800 whitespace-pre-line">
              {queryResult.answer.split('\n\n')[0]}
            </p>
            <ul className="list-disc list-inside text-gray-800 mt-3 space-y-1">
              {queryResult.answer
                .split('* ')
                .slice(1)
                .map((point, index) => (
                  <li key={index}>{point.trim()}</li>
                ))}
            </ul>
          </div>
        )}

        {queryResult && typeof queryResult === "object" && queryResult.sources?.length > 0 && (
          <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
            <h4 className="font-semibold text-yellow-700">📚 Sources:</h4>
            <ul className="mt-2 space-y-3">
              {queryResult.sources.map((src, i) => (
                <li
                  key={i}
                  className="bg-white p-3 rounded border border-yellow-200 shadow-sm"
                >
                  <div className="text-sm text-gray-700">
                    <strong>Source {i + 1}:</strong>{" "}
                    {src.source || src.filename || "Unknown"}
                    <br />
                    <em>Score: {src.score ? (src.score * 100).toFixed(1) + "%" : "N/A"}</em>
                    <br />
                    <small className="text-gray-600 block mt-1">
                      {src.content_preview || src.content || "No preview available"}
                    </small>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="bg-white rounded-xl p-4 shadow text-center">
      <div className="text-indigo-600 text-2xl font-bold">{value}</div>
      <div className="text-gray-500 text-sm mt-1">{label}</div>
    </div>
  );
}
