import { createContext, useContext, useReducer, ReactNode } from 'react'

interface SidebarContextType {
  isOpen: boolean
  setIsOpen: (isOpen: boolean) => void
  toggleSidebar: () => void
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined)

// Reducer function - stable and pure
type SidebarAction =
  | { type: 'TOGGLE' }
  | { type: 'SET', payload: boolean }

const sidebarReducer = (state: boolean, action: SidebarAction): boolean => {
  switch (action.type) {
    case 'TOGGLE':
      return !state
    case 'SET':
      return action.payload
    default:
      return state
  }
}

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [isOpen, dispatch] = useReducer(sidebarReducer, false)

  // These functions are now stable because dispatch is stable
  const toggleSidebar = () => dispatch({ type: 'TOGGLE' })
  const setIsOpen = (value: boolean) => dispatch({ type: 'SET', payload: value })

  return (
    <SidebarContext.Provider value={{ isOpen, setIsOpen, toggleSidebar }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebar() {
  const context = useContext(SidebarContext)
  if (context === undefined) {
    throw new Error('useSidebar must be used within a SidebarProvider')
  }
  return context
}
