import { Routes, Route, Navigate } from 'react-router-dom'
import { JobProvider } from './context/JobContext'
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
import OemFleetOverview from './pages/OemFleetOverview'
import OemModelHealth from './pages/OemModelHealth'
import OemEDA from './pages/OemEDA'
import OemWhatIf from './pages/OemWhatIf'
import OemRetrain from './pages/OemRetrain'
import AdminUsers from './pages/AdminUsers'

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const token = localStorage.getItem('ap_token')
  return token ? <>{children}</> : <Navigate to="/login" replace />
}

function OemRoute({ children }: { children: React.ReactNode }) {
  const role = localStorage.getItem('ap_role') ?? ''
  return role === 'OEM' || role === 'ADMIN'
    ? <>{children}</>
    : <Navigate to="/" replace />
}

function AdminRoute({ children }: { children: React.ReactNode }) {
  const role = localStorage.getItem('ap_role') ?? ''
  return role === 'ADMIN'
    ? <>{children}</>
    : <Navigate to="/" replace />
}

export default function App() {
  return (
    <JobProvider>
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
        {/* Dealer routes */}
        <Route index                element={<Dashboard />} />
        <Route path="vehicles/:vin" element={<VehicleDetail />} />
        <Route path="service-bay"   element={<ServiceBay />} />
        <Route path="inventory"     element={<Inventory />} />
        <Route path="alerts"        element={<Alerts />} />
        <Route path="driver-scores" element={<DriverScores />} />
        <Route path="workflows"     element={<Workflows />} />
        <Route path="upload"        element={<OemRoute><Upload /></OemRoute>} />

        {/* OEM-only routes */}
        <Route path="oem/fleet"   element={<OemRoute><OemFleetOverview /></OemRoute>} />
        <Route path="oem/models"  element={<OemRoute><OemModelHealth /></OemRoute>} />
        <Route path="oem/eda"     element={<OemRoute><OemEDA /></OemRoute>} />
        <Route path="oem/whatif"  element={<OemRoute><OemWhatIf /></OemRoute>} />
        <Route path="oem/retrain" element={<OemRoute><OemRetrain /></OemRoute>} />

        {/* Admin-only routes */}
        <Route path="admin/users" element={<AdminRoute><AdminUsers /></AdminRoute>} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
    </JobProvider>
  )
}
