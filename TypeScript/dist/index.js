"use strict";
console.log("Hello World");


/** @typedef { import ("../@types/person").Person } Person */

/**
 * @param {Person} person
 */
function printPerson(person) {
    console.log(person.name);
}