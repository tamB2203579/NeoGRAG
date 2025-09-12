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




// import React, { useEffect } from 'react'
// import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
// import WebFont from 'webfontloader'

// import Home from './pages/Home'
// import Auth from './pages/Auth/Auth'
// import Upload from './pages/Upload/Upload'
// import { AuthProvider } from './pages/Auth/AuthContext'
// import PrivateRoute from './pages/Auth/PrivateRouter'
// import './index.css'

// const App = () => {
//   useEffect(() => {
//     WebFont.load({
//       google: {
//         families: [
//           'K2D:vietnamese',
//           'Readex Pro:vietnamese'
//         ]
//       }
//     });
//   }, []);

//   return (
//     <AuthProvider>
//       <Router>
//         <Routes>
//           <Route path="/" element={<Home />} />
//           <Route path="/auth" element={<Auth />} />
//           <Route 
//             path="/upload" 
//             element={
//               <PrivateRoute>
//                 <Upload />
//               </PrivateRoute>
//             } 
//           />
//         </Routes>
//       </Router>
//     </AuthProvider>
//   )
// }

// export default App

import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import WebFont from 'webfontloader';

import Home from './pages/Home';
import Auth from './pages/Auth/Auth';
import Upload from './pages/Upload/Upload';
import { AuthProvider } from './pages/Auth/AuthContext';
import PrivateRoute from './pages/Auth/PrivateRouter';
import './index.css';

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
        {/* Client public route */}
        <Route path="/chat" element={<Home />} />

        {/* Admin routes */}
        <Route 
          path="/auth" 
          element={
            <AuthProvider>
              <Auth />
            </AuthProvider>
          } 
        />
        <Route 
          path="/upload" 
          element={
            <AuthProvider>
              <PrivateRoute>
                <Upload />
              </PrivateRoute>
            </AuthProvider>
          } 
        />
      </Routes>
    </Router>
  );
};

export default App;
