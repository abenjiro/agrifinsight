import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'
import { DashboardPage } from './pages/DashboardPage'
import { AnalysisPage } from './pages/AnalysisPage'
import { RecommendationsPage } from './pages/RecommendationsPage'
import { LoginPage } from './pages/LoginPage'
import { RegisterPage } from './pages/RegisterPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
        <Route path="/recommendations" element={<RecommendationsPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
      </Routes>
    </Layout>
  )
}

export default App

