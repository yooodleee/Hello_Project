// @types/person.d.ts

// 다음과 같은 모양을 갖는 객체 인터페이스
export interface Person {
    name: string;
    age: number;
}

// 기존 인터페이스를 확장하는 인터페이스
// JSDoc 주석으로는 이를 구현하기 힘들다.
export interface Student extends Person {
    semester: number;
}

// index.js
/** @typedef { import ("../@types/person").Person } Person */

/**
 * 첫 행의 주석은 타입스크립트에가 @/types/person에서 Person 형식을 임포트해서 Person이라는 
 * 이름으로 사용할 수 있도록 지시한다. 마치 string 같은 기본형을 이용하듯이 이 식별자로 함수 매개변수나
 * 객체에 주석을 추가할 수 있다.
 */