import React from "react";
import "./Auth.css";
import { assets } from "../../assets/assets";

export default function Auth() {
  return (
    <div className="auth-container">
      <div className="auth-box">
        <img src={assets.ctu} alt="Logo" className="auth-logo" />
        {/* <h2 className="auth-title">Hệ thống Quản lý tài liệu REBot</h2> */}

        <form className="auth-form">
          <div className="auth-input-group">
            <span class="material-symbols-outlined">mail</span>
            <input type="email" placeholder="Email" className="auth-input" />
          </div>

          <div className="auth-input-group">
          <span class="material-symbols-outlined">lock</span>
            <input type="password" placeholder="Mật khẩu" className="auth-input" />
          </div>

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
