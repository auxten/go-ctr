export const on = function(
  element: HTMLElement | Document | Window,
  event: string,
  handler: any, // EventListenerOrEventListenerObject
  useCapture = false,
): void {
  if (element && event && handler) {
    element.addEventListener(event, handler, useCapture)
  }
}

export const off = function(
  element: HTMLElement | Document | Window,
  event: string,
  handler: any, // EventListenerOrEventListenerObject
): void {
  if (element && event && handler) {
    element.removeEventListener(event, handler, false)
  }
}

/**
 * @param {HTMLElement} element
 * @param {string} className
 */
export function toggleClass(element, className) {
  if (!element || !className) {
    return
  }
  let classString = element.className
  const nameIndex = classString.indexOf(className)
  if (nameIndex === -1) {
    classString += `${className}`
  } else {
    classString =
      classString.substr(0, nameIndex) +
      classString.substr(nameIndex + className.length)
  }
  element.className = classString
}

/**
 * Check if an element has a class
 * @param {HTMLElement} elm
 * @param {string} cls
 * @returns {boolean}
 */
export function hasClass(ele, cls) {
  return !!ele.className.match(new RegExp(`(\\s|^)${cls}(\\s|$)`))
}

/**
 * Add class to element
 * @param {HTMLElement} elm
 * @param {string} cls
 */
export function addClass(ele, cls) {
  if (!hasClass(ele, cls)) ele.className += ` ${cls}`
}

/**
 * Remove class from element
 * @param {HTMLElement} elm
 * @param {string} cls
 */
export function removeClass(ele, cls) {
  if (hasClass(ele, cls)) {
    const reg = new RegExp(`(\\s|^)${  cls  }(\\s|$)`)
    ele.className = ele.className.replace(reg, ' ')
  }
}
