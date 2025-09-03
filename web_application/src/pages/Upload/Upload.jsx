import React, { useState } from "react";
import "./Upload.css";

export default function Upload() {
  const [files, setFiles] = useState([]);

  const handleFileChange = (e) => {
    setFiles([...files, ...Array.from(e.target.files)]);
  };

  const handleRemove = (index) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    setFiles(newFiles);
  };

  return (
    <div className="upload-container">
      <h2 className="upload-title">Hệ thống Quản lý tài liệu REBot</h2>

      <label className="upload-dropzone">
        <input
          type="file"
          accept="application/pdf"
          multiple
          onChange={handleFileChange}
          hidden
        />
        <div className="dropzone-content">
          <span class="material-symbols-outlined plus-icon">add_2</span>
          <p>Kéo & thả hoặc bấm để chọn file</p>
          <small>Dung lượng tối đa: 10MB</small>
        </div>
      </label>

      {/* File List */}
      {files.length > 0 && (
        <div className="file-list">
          {files.map((file, index) => (
            <div key={index} className="file-item">
              <div className="file-info">
                <span className="material-symbols-outlined">draft</span>
                <div>
                  <p className="file-name">{file.name}</p>
                  <small className="file-size">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </small>
                </div>
              </div>
              <div className="file-actions">
                <button
                  className="btn-icon"
                  onClick={() => handleRemove(index)}
                >
                  <span className="material-symbols-outlined">delete</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {files.length > 0 && (
        <button className="btn-upload">Tải lên</button>
      )}
    </div>
  );
}
