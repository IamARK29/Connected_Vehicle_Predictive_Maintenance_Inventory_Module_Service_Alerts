import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'
import Dashboard from './pages/Dashboard'
import VehicleDetail from './pages/VehicleDetail'
import ServiceBay from './pages/ServiceBay'
import Inventory from './pages/Inventory'
import Alerts from './pages/Alerts'
import DriverScores from './pages/DriverScores'
import Workflows from './pages/Workflows'
import Upload from './pages/Upload'

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const token = localStorage.getItem('ap_token')
  return token ? <>{children}</> : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <PrivateRoute>
            <Layout />
          </PrivateRoute>
        }
      >
        <Route index               element={<Dashboard />} />
        <Route path="vehicles/:vin" element={<VehicleDetail />} />
        <Route path="service-bay"  element={<ServiceBay />} />
        <Route path="inventory"    element={<Inventory />} />
        <Route path="alerts"       element={<Alerts />} />
        <Route path="driver-scores" element={<DriverScores />} />
        <Route path="workflows"    element={<Workflows />} />
        <Route path="upload"       element={<Upload />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
