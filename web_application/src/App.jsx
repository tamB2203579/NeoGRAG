// import React, { useEffect } from 'react'
// import Home from './pages/Home'
// import './index.css'
// import WebFont from 'webfontloader'

// const App = () => {
//   useEffect(() => {
//         WebFont.load({
//           google: {
//             families: [
//             'K2D:vietnamese',
//             'Readex Pro:vietnamese'
//           ]
//           }
//         });
//       }, []);
//   return (
//     <div>
//       <Home/>
//     </div>
//   );
// };

// export default App

import React, { useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import WebFont from 'webfontloader'

import Home from './pages/Home'
import Auth from './pages/Auth/Auth'
import Upload from './pages/Upload/Upload'
import './index.css'

const App = () => {
  useEffect(() => {
    WebFont.load({
      google: {
        families: [
          'K2D:vietnamese',
          'Readex Pro:vietnamese'
        ]
      }
    });
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/upload" element={<Upload />} />
      </Routes>
    </Router>
  )
}

export default App
