import { useEffect, useState } from 'react';
import { onSnapshot, doc, collection, getDoc, deleteDoc, query, orderBy } from 'firebase/firestore';
import { db } from '../../../firebase.config';
import WebFont from 'webfontloader';
import { assets } from '../../assets/assets';
import './Sidebar.css';

const Sidebar = ({ isOpen, onToggle, updateCurrentThread, updateChatHistory }) => {
  const [threads, setThreads] = useState([]);

  useEffect(() => {
    WebFont.load({
      google: {
        families: [
          'K2D:400,500,700&display=swap',
          'Readex Pro:400,500,700&display=swap',
        ],
      },
    });

    const threadsCollection = collection(db, 'threads');
    const q = query(threadsCollection, orderBy('timestamp', 'desc'));
    
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const threadsList = snapshot.docs.map((doc) => ({
        id: doc.id,
        title: doc.data().title || 'Cuộc trò chuyện mới',
        timestamp: doc.data().timestamp,
      }));
      setThreads(threadsList);
    }, (error) => {
      console.error('Lỗi tải danh sách thread:', error);
    });

    return () => unsubscribe();
  }, []);

  const handleDeleteThread = async (threadId) => {
    if (window.confirm('Bạn có chắc chắn muốn xóa cuộc trò chuyện này không?')) {
      try {
        await deleteDoc(doc(db, 'threads', threadId));

        updateCurrentThread(null);
        updateChatHistory([]);

        setThreads((prev) => prev.filter((thread) => thread.id !== threadId));
      } catch (error) {
        console.error('Lỗi xóa thread:', error);
      }
    }
  };

  const handleThreadClick = async (threadId) => {
    try {
      console.log(`Chuyển sang thread ${threadId}`);
      updateCurrentThread(threadId);

      const threadRef = doc(db, 'threads', threadId);
      const threadSnap = await getDoc(threadRef);

      if (threadSnap.exists()) {
        const threadData = threadSnap.data();
        updateChatHistory(threadData.contents || []);
      } else {
        console.warn('Không tìm thấy thread');
        updateChatHistory([]);
      }
    } catch (error) {
      console.error('Lỗi tải thread:', error);
      updateChatHistory([]);
    }
  };

  return (
    <div className={`sidebar ${isOpen ? 'open' : 'collapsed'}`}>
      <div className="top">
        <div className="menu-container">
          <button
            id="toggle-btn"
            className="material-symbols-outlined menu"
            onClick={onToggle} 
            style={{
              color: isOpen ? "#fff" : "#1f5ca9", 
              background: "transparent",
              border: "none",
              cursor: "pointer",
            }}
          >
            menu
          </button>

        </div>

        <img
          className="home_logo"
          src={assets.logo_icon}
          alt=""
          onClick={() => window.location.replace('/landing.html')}
        />

        <div className="new-chat" onClick={() => {
          updateCurrentThread(null);
          updateChatHistory([]);
        }}>
          <img src={assets.plus_icon} alt="" />
          <p>Cuộc trò chuyện mới</p>
        </div>

        <div className="recent">
          <p className="recent-title">Gần đây</p>
          <div className="recent-list">
            {threads.map((thread) => (
              <div
                className="recent-entry"
                key={thread.id}
                onClick={() => handleThreadClick(thread.id)}
              >
                <img src={assets.white_message_icon} alt="" />
                <p>{thread.title}</p>
                <img
                  src={assets.delete_icon}
                  alt='delete'
                  onClick={(e) => {
                    e.stopPropagation(); // prevent triggering parent click
                    handleDeleteThread(thread.id);
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      <div>
        <img
          className="home-btn"
          src={isOpen ? assets.white_home_icon : assets.home_icon}
          alt=""
          onClick={() => window.location.replace('/landing.html')}
        />
      </div>
    </div>
  );
};

export default Sidebar;