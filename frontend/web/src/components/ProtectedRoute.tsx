import { ReactNode, useEffect, useState } from 'react'
import { Navigate } from 'react-router-dom'
import api from '../services/api'

interface ProtectedRouteProps {
  children: ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('auth_token')

        // If no token exists, user is not authenticated
        if (!token) {
          setIsAuthenticated(false)
          setIsLoading(false)
          return
        }

        // Verify token with backend
        try {
          await api.get('/auth/verify')
          setIsAuthenticated(true)
        } catch (error: any) {
          // Token is invalid, clear it
          localStorage.removeItem('auth_token')
          localStorage.removeItem('user')
          setIsAuthenticated(false)
        }
      } catch (error) {
        console.error('Auth check failed:', error)
        // On error, clear tokens and mark as unauthenticated
        localStorage.removeItem('auth_token')
        localStorage.removeItem('user')
        setIsAuthenticated(false)
      } finally {
        setIsLoading(false)
      }
    }

    checkAuth()
  }, [])

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
          <p className="text-sm text-gray-600">Verifying authentication...</p>
        </div>
      </div>
    )
  }

  // If not authenticated, redirect to login
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  // If authenticated, render the protected content
  return <>{children}</>
}
