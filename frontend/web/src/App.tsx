import { Routes, Route } from 'react-router-dom'
import { DashboardLayout } from './components/DashboardLayout'
import { LandingLayout } from './components/LandingLayout'
import { ProtectedRoute } from './components/ProtectedRoute'
import { HomePage } from './pages/HomePage'
import { AboutUsPage } from './pages/AboutUsPage'
import ContactUsPage from './pages/ContactUsPage'
import { DashboardPage } from './pages/DashboardPage'
import { FarmsPage } from './pages/FarmsPage'
import FarmDetailPage from './pages/FarmDetailPage'
import { AIFeaturesPage } from './pages/AIFeaturesPage'
import { AnalysisPage } from './pages/AnalysisPage'
import { RecommendationsPage } from './pages/RecommendationsPage'
import { PlantingRecommendationsPage } from './pages/PlantingRecommendationsPage'
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
      <Route path="/ai-features" element={
        <LandingLayout>
          <AIFeaturesPage />
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
      <Route path="/about" element={
        <LandingLayout>
          <AboutUsPage />
        </LandingLayout>
      } />
      <Route path="/contact" element={
        <LandingLayout>
          <ContactUsPage />
        </LandingLayout>
      } />

      {/* Dashboard pages with DashboardLayout - Protected Routes */}
      <Route path="/dashboard" element={
        <ProtectedRoute>
          <DashboardLayout>
            <DashboardPage />
          </DashboardLayout>
        </ProtectedRoute>
      } />
      <Route path="/farms" element={
        <ProtectedRoute>
          <DashboardLayout>
            <FarmsPage />
          </DashboardLayout>
        </ProtectedRoute>
      } />
      <Route path="/farms/:id" element={
        <ProtectedRoute>
          <FarmDetailPage />
        </ProtectedRoute>
      } />
      <Route path="/farms/:farmId/planting" element={
        <ProtectedRoute>
          <DashboardLayout>
            <PlantingRecommendationsPage />
          </DashboardLayout>
        </ProtectedRoute>
      } />
      <Route path="/analysis" element={
        <ProtectedRoute>
          <DashboardLayout>
            <AnalysisPage />
          </DashboardLayout>
        </ProtectedRoute>
      } />
      <Route path="/recommendations" element={
        <ProtectedRoute>
          <DashboardLayout>
            <RecommendationsPage />
          </DashboardLayout>
        </ProtectedRoute>
      } />
    </Routes>
  )
}

export default App

