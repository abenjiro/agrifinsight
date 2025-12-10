import { Routes, Route, Outlet } from 'react-router-dom'
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
import { HarvestPredictionsPage } from './pages/HarvestPredictionsPage'
import { ReportsPage } from './pages/ReportsPage'
import { LoginPage } from './pages/LoginPage'
import { RegisterPage } from './pages/RegisterPage'
import { memo } from 'react'

// Wrapper components MUST be defined OUTSIDE App to prevent recreation on every render
// Using memo to prevent re-renders when App re-renders
const HomePageWrapper = memo(() => {
  return <LandingLayout><HomePage /></LandingLayout>
})
HomePageWrapper.displayName = 'HomePageWrapper'

const AIFeaturesPageWrapper = memo(() => {
  return <LandingLayout><AIFeaturesPage /></LandingLayout>
})
AIFeaturesPageWrapper.displayName = 'AIFeaturesPageWrapper'

const LoginPageWrapper = memo(() => {
  return <LandingLayout><LoginPage /></LandingLayout>
})
LoginPageWrapper.displayName = 'LoginPageWrapper'

const RegisterPageWrapper = memo(() => {
  return <LandingLayout><RegisterPage /></LandingLayout>
})
RegisterPageWrapper.displayName = 'RegisterPageWrapper'

const AboutUsPageWrapper = memo(() => {
  return <LandingLayout><AboutUsPage /></LandingLayout>
})
AboutUsPageWrapper.displayName = 'AboutUsPageWrapper'

const ContactUsPageWrapper = memo(() => {
  return <LandingLayout><ContactUsPage /></LandingLayout>
})
ContactUsPageWrapper.displayName = 'ContactUsPageWrapper'

function App() {
  return (
    <Routes>
      {/* Landing pages with Header and Footer */}
      <Route path="/" element={<HomePageWrapper />} />
      <Route path="/ai-features" element={<AIFeaturesPageWrapper />} />
      <Route path="/login" element={<LoginPageWrapper />} />
      <Route path="/register" element={<RegisterPageWrapper />} />
      <Route path="/about" element={<AboutUsPageWrapper />} />
      <Route path="/contact" element={<ContactUsPageWrapper />} />

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
      <Route path="/crops/:cropId/harvest" element={
        <ProtectedRoute>
          <DashboardLayout>
            <HarvestPredictionsPage />
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
      <Route path="/reports" element={
        <ProtectedRoute>
          <DashboardLayout>
            <ReportsPage />
          </DashboardLayout>
        </ProtectedRoute>
      } />
    </Routes>
  )
}

export default App

