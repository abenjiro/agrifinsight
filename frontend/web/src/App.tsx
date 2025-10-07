import { Routes, Route } from 'react-router-dom'
import { DashboardLayout } from './components/DashboardLayout'
import { LandingLayout } from './components/LandingLayout'
import { HomePage } from './pages/HomePage'
import { DashboardPage } from './pages/DashboardPage'
import { AnalysisPage } from './pages/AnalysisPage'
import { RecommendationsPage } from './pages/RecommendationsPage'
import { LoginPage } from './pages/LoginPage'
import { RegisterPage } from './pages/RegisterPage'

function App() {
  return (
    <Routes>
      {/* Landing pages with Header and Footer */}
      <Route path="/" element={
        <LandingLayout>
          <HomePage />
        </LandingLayout>
      } />
      <Route path="/login" element={
        <LandingLayout>
          <LoginPage />
        </LandingLayout>
      } />
      <Route path="/register" element={
        <LandingLayout>
          <RegisterPage />
        </LandingLayout>
      } />
      
      {/* Dashboard pages with DashboardLayout */}
      <Route path="/dashboard" element={
        <DashboardLayout>
          <DashboardPage />
        </DashboardLayout>
      } />
      <Route path="/analysis" element={
        <DashboardLayout>
          <AnalysisPage />
        </DashboardLayout>
      } />
      <Route path="/recommendations" element={
        <DashboardLayout>
          <RecommendationsPage />
        </DashboardLayout>
      } />
    </Routes>
  )
}

export default App

