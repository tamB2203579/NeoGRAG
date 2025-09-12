import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { collection, query, where, getDocs } from "firebase/firestore";
import { db } from "../../../firebase.config";
import "./Auth.css";
import { assets } from "../../assets/assets";
import { useAuth } from "./AuthContext";

export default function Auth() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    try {
      const q = query(
        collection(db, "admin"),
        where("email", "==", email),
        where("password", "==", password)
      );
      const querySnapshot = await getDocs(q);

      if (!querySnapshot.empty) {
        login();
        navigate("/upload");
      } else {
        setError("Email hoặc mật khẩu không đúng!");
        setTimeout(() => setError(""), 3000);
      }
    } catch (err) {
      console.error("Login error: ", err);
      setError("Có lỗi xảy ra, vui lòng thử lại.");
      setTimeout(() => setError(""), 3000);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <img src={assets.ctu} alt="Logo" className="auth-logo" />

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="auth-input-group">
            <span className="material-symbols-outlined">mail</span>
            <input
              type="email"
              placeholder="Email"
              className="auth-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div className="auth-input-group">
            <span className="material-symbols-outlined">lock</span>
            <input
              type={showPassword ? "text" : "password"}
              placeholder="Mật khẩu"
              className="auth-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <span
              className="material-symbols-outlined"
              onClick={() => setShowPassword(!showPassword)}
              style={{ cursor: "pointer" }}
            >
              {showPassword ? "visibility_off" : "visibility"}
            </span>
          </div>

          {error && (
            <p style={{ color: "red", fontSize: "17px", marginBottom: "10px" }}>
              {error}
            </p>
          )}

          <div className="auth-options">
            <a href="#">Quên mật khẩu?</a>
          </div>

          <button type="submit" className="auth-button">
            Đăng nhập
          </button>
        </form>
      </div>
    </div>
  );
}
